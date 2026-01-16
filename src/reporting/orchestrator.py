import os
import logging
from .generator import NotebookGenerator

class Phase4Orchestrator:
    def __init__(self, project_root):
        """
        Initialize the Phase 4 Orchestrator.
        
        Args:
            project_root (str): Absolute path to the project root directory.
        """
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        self.generator = NotebookGenerator()
        
        # Define the templates relative to project root
        self.templates = {
            "4_1_methodology": "academic/methodological_report/phase4_4_1_metodologia_eda.ipynb",
            "4_2_math": "academic/methodological_report/phase4_4_2_matematicas_eda.ipynb",
            "4_3_results": "academic/methodological_report/phase4_4_3_resultados_eda.ipynb",
            "4_4_interpretation": "academic/methodological_report/phase4_4_4_interpretacion_eda.ipynb",
            "general_report": "academic/Reporte_Integral_TFM (Actualizado).ipynb"
        }

    def generate_reports(self, phase3_csv_path, output_dir_base):
        """
        Generates all Phase 4 notebooks using the provided Phase 3 data.
        
        Args:
            phase3_csv_path (str): Absolute path to phase3_results.csv
            output_dir_base (str): Absolute path to the directory where reports should be saved.
        """
        # Convert to absolute paths to avoid issues with notebook execution CWD
        phase3_csv_path = os.path.abspath(phase3_csv_path)
        output_dir_base = os.path.abspath(output_dir_base)
        
        self.logger.info(f"Starting Phase 4 report generation. Output: {output_dir_base}")
        
        if not os.path.exists(phase3_csv_path):
            self.logger.error(f"Phase 3 CSV not found at {phase3_csv_path}")
            raise FileNotFoundError(f"Phase 3 CSV not found at {phase3_csv_path}")

        # Ensure output directory exists
        os.makedirs(output_dir_base, exist_ok=True)
        
        # Deduce other files based on standard structure (same dir as phase3 results)
        phase3_dir = os.path.dirname(phase3_csv_path)
        
        # Artifacts directory
        artifacts_dir = os.path.join(phase3_dir, "artifacts")
        
        # Subspaces directory for General Report
        subspaces_dir = os.path.join(phase3_dir, "artifacts", "subspaces")
        if not os.path.exists(subspaces_dir):
             # Try fallback
             subspaces_dir = os.path.join(self.project_root, "data", "phase3", "artifacts", "subspaces")

        # Prepare replacements
        # We need to wrap paths in quotes because they will be injected directly into code
        replacements = {
            "PHASE3_CSV": f"'{phase3_csv_path}'",
            "CSV_PATH": f"'{phase3_csv_path}'",
            "base_dir": f"'{subspaces_dir}'",
            "ARTIFACTS_DIR": f"'{artifacts_dir}'",
            # Default empty/placeholder if not valid, but we try to find them
        }
        
        # Check for anchors
        # Priority: Phase 3 dir (subdirectories or direct), then Data dir
        # Common locations: 
        # - data/phase3/anchors_matrix.csv
        # - data/phase3/artifacts/anchors/anchors_matrix.csv
        # - data/phase3/artifacts/embeddings_anchors.csv
        possible_anchors_names = ["anchors_matrix.csv", "anchors.csv", "embeddings_anchors.csv"]
        possible_anchors_dirs = [
            phase3_dir, 
            os.path.join(phase3_dir, "artifacts"),
            os.path.join(phase3_dir, "artifacts", "anchors"),
            os.path.join(phase3_dir, "anchors"),
            os.path.join(self.project_root, "data")
        ]
        
        found_anchor = False
        for d in possible_anchors_dirs:
            if found_anchor: break
            if not os.path.exists(d): continue
            for name in possible_anchors_names:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    replacements["ANCHORS_CSV"] = f"'{p}'"
                    self.logger.info(f"Found ANCHORS_CSV: {p}")
                    found_anchor = True
                    break
        
        # Check for dimensions
        # Priority: Phase 3 dir, then Data dir
        possible_dims = ["dimensions.json", "dimensiones_ancla.json"]
        found_dim = False
        possible_dims_dirs = [phase3_dir, os.path.join(self.project_root, "data")]
        
        for d in possible_dims_dirs:
            if found_dim: break
            if not os.path.exists(d): continue
            for name in possible_dims:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    replacements["DIMENSIONS_JSON"] = f"'{p}'"
                    self.logger.info(f"Found DIMENSIONS_JSON: {p}")
                    found_dim = True
                    break

        # Check for manifest
        manifest_path = os.path.join(phase3_dir, "manifest.json")
        if os.path.exists(manifest_path):
            replacements["MANIFEST_JSON"] = f"'{manifest_path}'"
            self.logger.info(f"Found MANIFEST_JSON: {manifest_path}")

        # Subspaces directory for General Report
        # Ensure output directory for images exists if needed
        artifacts_path_output = os.path.join(output_dir_base, "artifacts")
        os.makedirs(artifacts_path_output, exist_ok=True)
        # Do NOT overwrite ARTIFACTS_DIR as it is used for input retrieval
        # replacements["ARTIFACTS_DIR"] = f"'{artifacts_path_output}'"

        # Inject PROJECT_ROOT for imports
        # We replace the hardcoded relative path with the absolute project root
        replacements["sys.path.append('..')"] = f"sys.path.append('{self.project_root}')"

        # Generate each notebook
        for key, relative_template_path in self.templates.items():
            template_full_path = os.path.join(self.project_root, relative_template_path)
            
            if not os.path.exists(template_full_path):
                self.logger.warning(f"Template not found: {template_full_path}. Skipping.")
                continue

            # Determine output filename
            # For general report, rename it. For others, keep original name.
            if key == "general_report":
                output_filename = "General_Report.ipynb"
            else:
                output_filename = os.path.basename(relative_template_path)
                
            output_full_path = os.path.join(output_dir_base, output_filename)
            
            self.logger.info(f"Generating {key} -> {output_full_path}")
            
            try:
                self.generator.generate_and_execute(
                    template_path=template_full_path,
                    output_path=output_full_path,
                    replacements=replacements
                )
                self.logger.info(f"Successfully generated {output_filename}")
            except Exception as e:
                self.logger.error(f"Failed to generate {output_filename}: {e}")
                # We continue to the next one instead of crashing everything?
                # For a pipeline, maybe we should raise? 
                # Let's log error but allow others to proceed, then raise at end if critical?
                # For now, just log error.
