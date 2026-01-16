import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import logging

class NotebookGenerator:
    def __init__(self, timeout=600, kernel_name='python3'):
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.logger = logging.getLogger(__name__)

    def generate_and_execute(self, template_path, output_path, replacements):
        """
        Generates a new notebook from a template by replacing variable values and executes it.
        
        Args:
            template_path (str): Path to the source notebook template.
            output_path (str): Path where the generated notebook will be saved.
            replacements (dict): Dictionary of variable names (str) and their new values (str).
                                 Values should be Python code strings (e.g., '"path/to/file.csv"').
        """
        self.logger.info(f"Generating notebook from {template_path} to {output_path}")
        
        # 1. Read the notebook
        with open(template_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        # 2. Inject variables
        # We look for the first code cell or a specific cell to inject parameters.
        # Strategy: Prepend a new cell with the injected variables to ensure they override anything valid.
        # Alternatively, we can replace the specific lines if we want to be cleaner, but prepending is robust
        # for "parameterizing" notebooks without a strict tagging system like papermill.
        # However, since the notebooks have hardcoded paths at the top, we want to replace those.
        # Let's try to match lines starting with the variable name.
        
        self._inject_variables(nb, replacements)

        # 3. Execute the notebook
        self.logger.info("Executing notebook...")
        ep = ExecutePreprocessor(timeout=self.timeout, kernel_name=self.kernel_name)
        
        try:
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(output_path)}})
        except Exception as e:
            self.logger.error(f"Error executing notebook {output_path}: {e}")
            raise

        # 4. Save the executed notebook
        self.logger.info(f"Saving executed notebook to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    def _inject_variables(self, nb, replacements):
        # We'll inject a new cell at the top with the replacements. 
        # This is the safest way to ensure our values take precedence, 
        # provided the notebook doesn't overwrite them later with hardcoded values *without* checking if they exist.
        # BUT, the existing notebooks DO have hardcoded values like `PHASE3_CSV = "..."`.
        # So prepending `PHASE3_CSV = "new_path"` works IF the original code is conditional or if we remove the original lines.
        # Since we are automating this, let's try to replace the content if we find the definition, 
        # OR just prepend a cell and assume python will use the latest definition? 
        # actually, if we prepend, the original code `PHASE3_CSV = ...` will execute LATER and overwrite our injection.
        
        # So we MUST APPEND a cell AFTER imports but BEFORE usage? 
        # Or simpler: Iterating cells and replacing the lines.
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                lines = cell.source.split('\n')
                new_lines = []
                for line in lines:
                    replaced = False
                    for var_name, var_value in replacements.items():
                        # Case 1: Variable assignment
                        is_assignment = line.strip().startswith(f"{var_name} =") or line.strip().startswith(f"{var_name}=")
                        # Case 2: Exact line match (for statements like sys.path.append)
                        is_match = line.strip() == var_name
                        
                        if is_assignment or is_match:
                            # Capture leading whitespace
                            indent = line[:len(line) - len(line.lstrip())]
                            
                            # Comment out the old line
                            new_lines.append(f"{indent}# {line.strip()} # Replaced by generator")
                            
                            # Add the new line with indentation
                            if is_assignment:
                                new_line = f"{indent}{var_name} = {var_value}"
                                new_lines.append(new_line)
                                self.logger.info(f"Replaced {var_name} with {var_value}")
                            else:
                                new_lines.append(f"{indent}{var_value}")
                                self.logger.info(f"Replaced line matching {var_name}")
                            replaced = True
                            break
                    if not replaced:
                        new_lines.append(line)
                cell.source = '\n'.join(new_lines)
