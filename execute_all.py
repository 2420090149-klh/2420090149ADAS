import os
import json
import subprocess

notebooks = [f for f in os.listdir('.') if f.endswith('.ipynb')]

for nb in notebooks:
    print(f"Processing {nb}...")
    py_file = nb.replace('.ipynb', '.py')
    try:
        with open(nb, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        code_cells = [cell['source'] for cell in data.get('cells', []) if cell.get('cell_type') == 'code']
        code = ""
        for cell_source in code_cells:
            # Handle lines
            if isinstance(cell_source, list):
                # Filter out google.colab and other problematic magic commands
                filtered_source = [line for line in cell_source if not line.startswith('!') and not line.startswith('%') and 'google.colab' not in line]
                code += "".join(filtered_source) + "\n\n"
            else:
                code += cell_source + "\n\n"
                
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write(code)
            
        print(f"Executing {py_file}...")
        # Run with timeout to prevent infinite hangs (e.g. cv2.waitKey)
        subprocess.run(['python', py_file], timeout=15)
        print(f"Successfully executed {py_file}")
    except Exception as e:
        print(f"Failed to execute {nb}: {e}")
