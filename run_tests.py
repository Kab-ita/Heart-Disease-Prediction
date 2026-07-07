import subprocess
import sys


def run_step(command, description, log_file=None):
    print(f"\n--- Running: {description} ---")
    try:
        if log_file:
            with open(log_file, "w", encoding="utf-8") as f:
                subprocess.run(command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
            print(f"Done. Results saved to {log_file}")
        else:
            subprocess.run(command, shell=True, check=True)
            print("Done.")
    except subprocess.CalledProcessError:
        print(f"Error occurred during: {description}")
        sys.exit(1)


if __name__ == "__main__":  
    run_step("python train.py", "ML Model Training Pipeline") 
    run_step("pytest test_ml_logic.py", "Unit Testing", log_file="unit_test_report.txt") 
    run_step("python manage.py test predictor", "Django Integration Testing", log_file="integration_test_report.txt")
    print("\nAll phases completed! Your test reports are ready in your project root folder.")