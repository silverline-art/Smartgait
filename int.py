import sys

def main():
    print("Select the type of data to analyze:")
    print("a) PE (Pose Estimation)")
    print("b) SI (Smart Insole)")
    choice = input("Enter your choice (a/b): ").strip().lower()

    if choice not in ('a', 'b'):
        print("Invalid choice. Please enter 'a' or 'b'.")
        sys.exit(1)

    # Temporary default path (can be changed as needed)
    default_path = "/path/to/your/data.csv"
    file_path = input(f"Enter the path to the data file [{default_path}]: ").strip()
    if not file_path:
        file_path = default_path

    if choice == 'a':
        print(f"Running Pose Estimation analysis on {file_path}...")
        # Import and call your PE analysis function here
        # from raw_codes.gait_analysis import main as pe_main
        # pe_main(file_path)
    elif choice == 'b':
        print(f"Running Smart Insole analysis on {file_path}...")
        # Import and call your SI analysis function here
        # from raw_codes.smart_insole_analysis import main as si_main
        # si_main(file_path)

if __name__ == "__main__":
    main()