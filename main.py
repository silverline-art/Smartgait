import os
from config import SELECT_TASK_INDEX, output_path

plot_dir = os.path.join(output_path, "plot")
os.makedirs(plot_dir, exist_ok=True)

TASKS = ["PE", "SE"]
# Check if the selected index is valid before assignment
if 0 <= SELECT_TASK_INDEX < len(TASKS):
    Selected_Task = TASKS[SELECT_TASK_INDEX]
    if Selected_Task == "PE":
        # Run the Pose Estimation pipeline
        import PE_pipeline
        PE_pipeline.main()
    elif Selected_Task == "SE":
        # Run the Smart Insole pipeline
        import SE_pipeline
        SE_pipeline.main()
    else:
        print("Selected task is not implemented.")
else:
    print("""
            Invalid task selected!
            Please check your configuration.
    """)

