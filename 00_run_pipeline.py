import os
import subprocess
import sys

def run_script(script_name, python_path=sys.executable):
    """Runs a python script and waits for it to finish."""
    print(f"\n{'='*50}")
    print(f"Running: {script_name}")
    print(f"{'='*50}")
    
    try:
        # Use subprocess.run to execute the script
        result = subprocess.run([python_path, script_name], check=True)
        if result.returncode == 0:
            print(f"Successfully finished: {script_name}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    # Base directory is the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the scripts in order
    scripts = [
        "a_scrape_formations.py",
        "b_get_market_values.py",
        "c_get_fifa_ratings.py",
        "d_get_attendance_and_position.py",
        "e_scrape_xg.py",
        "f_get_form.py",
        "g_apply_pca.py",
        "h_get_predictions.py"
    ]
    
    # Define paths to python executables
    # Most scripts run in the current environment (base)
    # The NN prediction script (h) requires the fbref environment
    base_python = sys.executable
    fbref_python = "/Users/yasharjam/anaconda3/envs/fbref/bin/python"
    
    for script in scripts:
        script_path = os.path.join(base_dir, script)
        
        # Select the correct environment
        if script == "h_get_predictions.py":
            current_python = fbref_python
        else:
            current_python = base_python
            
        success = run_script(script_path, current_python)
        
        if not success:
            print(f"\nCRITICAL ERROR: Pipeline stopped at {script}")
            sys.exit(1)
            
    print("\n" + "#"*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("#"*50)

if __name__ == "__main__":
    main()
