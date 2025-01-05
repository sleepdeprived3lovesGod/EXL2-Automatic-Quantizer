import os
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import configparser
from datetime import datetime

# Function to check and update the configuration file version
def check_config_version():
    config = configparser.ConfigParser()
    config_file = 'config.ini'

    # Check if the config file exists
    if os.path.exists(config_file):
        config.read(config_file)

        # Check if the 'Settings' section exists
        if 'Settings' not in config:
            config['Settings'] = {}

        # Get the current version number
        current_version = config['Settings'].get('version', '0.0')

        # Compare the current version with the required version
        if float(current_version) < 3.22:
            # Delete the existing config file
            os.remove(config_file)
            config['Settings'] = {}
            config['Settings']['version'] = '3.22'
            with open(config_file, 'w') as configfile:
                config.write(configfile)
            messagebox.showinfo("Config Updated", "The configuration file has been updated to version 3.22.")
        else:
            # Ensure the version number is set in the config
            config['Settings']['version'] = '3.22'
            with open(config_file, 'w') as configfile:
                config.write(configfile)
    else:  # Create a new config file with version 3.22
        config['Settings'] = {}
        config['Settings']['version'] = '3.22'
        with open(config_file, 'w') as configfile:
            config.write(configfile)
        messagebox.showinfo("Config Created", "A new configuration file has been created with version 3.22.")

# Load configuration from config.ini
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Ensure the 'Settings' section exists
    if 'Settings' not in config:
        config['Settings'] = {}

    # Retrieve configuration values with default values if not present
    values = {
        'model_name': config['Settings'].get('model_name', ''),
        'raw_weights_dir': config['Settings'].get('raw_weights_dir', ''),
        'bits_per_head': config['Settings'].get('bits_per_head', '6'),
        'bpw_values': config['Settings'].get('bpw_values', ''),
        'venv_path': config['Settings'].get('venv_path', ''),
        'author_name': config['Settings'].get('author_name', ''),
        'exllamav2_dir': config['Settings'].get('exllamav2_dir', ''),
        'cuda_device': config['Settings'].get('cuda_device', '0'),
        'version': config['Settings'].get('version', '3.22')  # Ensure version is included
    }
    return values

# Save configuration to config.ini
def save_config(values):
    config = configparser.ConfigParser()
    config['Settings'] = values
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

# Function to run quantization on a specific GPU
def run_quantization_on_gpu(model_name, raw_weights_dir, bits_per_head, bpw, venv_path, author_name, cuda_device, exllamav2_dir, job_dir, measurement_json_path):
    try:
        # Construct the bpw string (e.g., "4bpw")
        bpw_str = f"{bpw}bpw"

        # Construct the temporary quantization directory
        temp_quant_dir = os.path.join(job_dir, f"temp_{model_name}_{bpw_str}_H{bits_per_head}")

        # Construct the final output directory with _EXL2_ in the name
        output_dir = os.path.join(job_dir, f"{model_name}_EXL2_{bpw_str}_H{bits_per_head}")

        # If an author name is provided, include it in the output directory name
        if author_name:
            output_dir = os.path.join(job_dir, f"{author_name}_{model_name}_EXL2_{bpw_str}_H{bits_per_head}")

        # Remove existing temp quant directory if it exists
        if os.path.exists(temp_quant_dir):
            shutil.rmtree(temp_quant_dir)

        # Create new temp quant directory
        os.makedirs(temp_quant_dir)

        # Remove existing output directory if it exists
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Create new output directory
        os.makedirs(output_dir)

        # Get the activate command based on the operating system
        activate_command = get_activate_command(venv_path)

        # Construct the path to the convert.py script
        convert_script_path = os.path.join(exllamav2_dir, 'convert.py')

        # Construct the quantization command
        quant_command = (
            f'{activate_command} && '
            f'set CUDA_VISIBLE_DEVICES={cuda_device} && '
            f'python {convert_script_path} '
            f'-i "{raw_weights_dir}" '
            f'-o "{temp_quant_dir}" '
            f'-m "{measurement_json_path}" '
            f'-cf "{output_dir}" '
            f'-b {bpw} '
            f'-hb {bits_per_head}'
        )

        # Adjust the command for non-Windows systems
        if os.name != 'nt':
            quant_command = quant_command.replace('set ', '')

        # Run the quantization command
        subprocess.run(quant_command, shell=True, check=True)

        # Clean up temporary directories
        if os.path.exists(temp_quant_dir):
            shutil.rmtree(temp_quant_dir)

        # Copy measurement.json to the output directory
        shutil.copy(measurement_json_path, output_dir)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Command Error", f"An error occurred while running a command on GPU {cuda_device}: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred on GPU {cuda_device}: {e}")
    finally:
        # Update the progress bar
        progress_bar.step(1)
        root.update_idletasks()

# Run commands in a separate thread
def run_commands_thread():
    try:  # Save the current values to the config file first
        values = {
            'model_name': model_name_entry.get(),
            'raw_weights_dir': raw_weights_entry.get(),
            'bits_per_head': ','.join([bits_per_head.get() for bits_per_head in bits_per_head_vars if bits_per_head.get() != ""]),
            'bpw_values': ','.join([bpw.get() for bpw in bpw_vars if bpw.get() != ""]) + ',' + ','.join(custom_bpw_entry.get().strip().split(',')),
            'venv_path': venv_path_entry.get(),
            'author_name': author_name_entry.get(),
            'exllamav2_dir': exllamav2_dir_entry.get(),
            'cuda_device': cuda_device_entry.get(),
            'version': '3.22'  # Ensure version is included
        }
        save_config(values)  # Save configuration

        # Get user inputs
        model_name = values['model_name']
        raw_weights_dir = values['raw_weights_dir']
        bits_per_head_values = [bits_per_head.get() for bits_per_head in bits_per_head_vars if bits_per_head.get() != ""] if values['bits_per_head'] else []
        venv_path = values['venv_path']
        author_name = values['author_name']
        exllamav2_dir = values['exllamav2_dir']
        cuda_device = values['cuda_device']

        # Get selected bpw values from checkboxes and custom entry
        bpw_values = [bpw.get() for bpw in bpw_vars if bpw.get() != ""] if values['bpw_values'] else []
        custom_bpw_values = custom_bpw_entry.get().strip().split(',')
        custom_bpw_values = [bpw.strip() for bpw in custom_bpw_values if bpw.strip()]

        # Combine bpw values
        bpw_values.extend(custom_bpw_values)

        # Check if bpw_values is empty after combining
        if not bpw_values:
            messagebox.showerror("Input Error", "Please select or enter at least one bpw value.")
            return

        # Check if bits_per_head_values is empty
        if not bits_per_head_values:
            messagebox.showerror("Input Error", "Please select at least one bits per head value.")
            return

        # Create job directory based on model name and current time
        current_time = datetime.now().strftime("%H%M%S")
        job_dir = os.path.join(os.getcwd(), f"{model_name}_{current_time}")
        os.makedirs(job_dir, exist_ok=True)

        # Step 2: Set up directories
        temp_measurement_dir = os.path.join(job_dir, "temp_measurement")

        # Remove existing temp directory if it exists
        if os.path.exists(temp_measurement_dir):
            shutil.rmtree(temp_measurement_dir)

        # Create new temp directory
        os.makedirs(temp_measurement_dir)

        # Prepare output directories
        for bits_per_head in bits_per_head_values:
            for bpw in bpw_values:
                bpw_str = f"{bpw}bpw"
                output_dir = os.path.join(job_dir, f"{model_name}_EXL2_{bpw_str}_H{bits_per_head}")

                # If an author name is provided, include it in the output directory name
                if author_name:
                    output_dir = os.path.join(job_dir, f"{author_name}_{model_name}_EXL2_{bpw_str}_H{bits_per_head}")

                # Create the output directory
                os.makedirs(output_dir, exist_ok=True)

        # Step 3: Check if measurement.json is provided
        measurement_path = measurement_path_entry.get()

        if measurement_path and os.path.exists(measurement_path):
            # Check if measurement.json is already in the correct location
            if os.path.abspath(measurement_path) != os.path.abspath(os.path.join(job_dir, "measurement.json")):
                # Copy measurement.json to the job directory
                shutil.copy(measurement_path, os.path.join(job_dir, "measurement.json"))
        else:
            # First command to take measurements
            activate_command = get_activate_command(venv_path)
            convert_script_path = os.path.join(exllamav2_dir, 'convert.py')
            measurement_command = (
                f'{activate_command} && '
                f'set CUDA_VISIBLE_DEVICES={cuda_device} && '
                f'python {convert_script_path} '
                f'-i "{raw_weights_dir}" '
                f'-o "{temp_measurement_dir}" '
                f'-om "{os.path.join(job_dir, "measurement.json")}"'
            )

            # Adjust the command for non-Windows systems
            if os.name != 'nt':
                measurement_command = measurement_command.replace('set ', '')

            # Run the measurement command
            subprocess.run(measurement_command, shell=True, check=True)

        # Initialize progress bar
        total_tasks = len(bits_per_head_values) * len(bpw_values)
        progress_bar['maximum'] = total_tasks
        progress_bar.grid(row=12, column=1, columnspan=2, padx=10, pady=10)
        progress_bar.start(500)  # Start the indeterminate progress bar with a smooth speed

        # Run quantization tasks sequentially
        measurement_json_path = os.path.join(job_dir, "measurement.json")
        for bits_per_head in bits_per_head_values:
            for bpw in bpw_values:
                run_quantization_on_gpu(model_name, raw_weights_dir, bits_per_head, bpw, venv_path, author_name, cuda_device, exllamav2_dir, job_dir, measurement_json_path)

        # Remove measurement.json from the job directory
        if os.path.exists(os.path.join(job_dir, "measurement.json")):
            os.remove(os.path.join(job_dir, "measurement.json"))

        # Remove temp_measurement directory
        if os.path.exists(temp_measurement_dir):
            shutil.rmtree(temp_measurement_dir)

        # Show success message
        messagebox.showinfo("Success", "All processes completed.")

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Command Error", f"An error occurred while running a command: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    finally:
        # Stop the progress bar
        progress_bar.stop()
        progress_bar.grid_forget()

        # Enable the Run button
        run_button.config(state="normal")

        # Re-enable form controls
        enable_form_controls()

# Get activate command based on OS
def get_activate_command(venv_path):
    if os.name == 'nt':  # Windows
        return f'"{venv_path}\\Scripts\\activate.bat"'
    else:  # Unix or Linux
        return f'source "{venv_path}/bin/activate"'

# Browse weights directory
def browse_weights():
    folder_selected = filedialog.askdirectory()
    if folder_selected:  # Check if a directory was selected
        raw_weights_entry.delete(0, tk.END)
        raw_weights_entry.insert(0, folder_selected)

# Browse virtual environment directory
def browse_venv():
    folder_selected = filedialog.askdirectory()
    if folder_selected:  # Check if a directory was selected
        venv_path_entry.delete(0, tk.END)  # Clear the entry
        venv_path_entry.insert(0, folder_selected)  # Update the entry with the selected path

# Browse measurement.json file
def browse_measurement():
    file_selected = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_selected:  # Check if a file was selected
        measurement_path_entry.delete(0, tk.END)  # Clear the entry
        measurement_path_entry.insert(0, file_selected)  # Update the entry with the selected file path

# Browse exllamav2 directory
def browse_exllamav2():
    folder_selected = filedialog.askdirectory()
    if folder_selected:  # Check if a directory was selected
        exllamav2_dir_entry.delete(0, tk.END)  # Clear the entry
        exllamav2_dir_entry.insert(0, folder_selected)  # Update the entry with the selected path

# Start the run commands in a separate thread
def start_run_commands():
    run_button.config(state="disabled")  # Disable the Run button

    # Disable form controls
    disable_form_controls()

    # Show the progress bar
    progress_bar.grid(row=12, column=1, columnspan=2, padx=10, pady=10)  # Display the progress bar
    progress_bar.start(500)  # Start the indeterminate progress bar with a smooth speed

    # Start the run commands in a separate thread
    thread = threading.Thread(target=run_commands_thread)
    thread.start()

# Disable form controls
def disable_form_controls():
    model_name_entry.config(state="disabled")
    raw_weights_entry.config(state="disabled")
    raw_weights_button.config(state="disabled")
    for bits_per_head_checkbox in bits_per_head_checkboxes:
        bits_per_head_checkbox.config(state="disabled")
    for bpw_checkbox in bpw_checkboxes:
        bpw_checkbox.config(state="disabled")
    custom_bpw_entry.config(state="disabled")
    measurement_path_entry.config(state="disabled")
    measurement_path_button.config(state="disabled")
    venv_path_entry.config(state="disabled")
    venv_path_button.config(state="disabled")
    author_name_entry.config(state="disabled")
    cuda_device_entry.config(state="disabled")
    exllamav2_dir_entry.config(state="disabled")
    exllamav2_dir_button.config(state="disabled")

# Enable form controls
def enable_form_controls():
    model_name_entry.config(state="normal")
    raw_weights_entry.config(state="normal")
    raw_weights_button.config(state="normal")
    for bits_per_head_checkbox in bits_per_head_checkboxes:
        bits_per_head_checkbox.config(state="normal")
    for bpw_checkbox in bpw_checkboxes:
        bpw_checkbox.config(state="normal")
    custom_bpw_entry.config(state="normal")
    measurement_path_entry.config(state="normal")
    measurement_path_button.config(state="normal")
    venv_path_entry.config(state="normal")
    venv_path_button.config(state="normal")
    author_name_entry.config(state="normal")
    cuda_device_entry.config(state="normal")
    exllamav2_dir_entry.config(state="normal")
    exllamav2_dir_button.config(state="normal")

# Check and update the configuration file version
check_config_version()

# Load configuration
config_values = load_config()

# Create the main window
root = tk.Tk()
root.title("Automatic Quantizer")

# Create a style for the buttons
style = ttk.Style()
style.configure('Nice.TButton', padding=(5, 5))

# Model Name
model_name_label = ttk.Label(root, text="Model Name:", anchor="e")
model_name_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
model_name_entry = ttk.Entry(root, width=50)
model_name_entry.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky="w")
model_name_entry.insert(0, config_values.get('model_name', ''))

# Raw Weights Directory
raw_weights_label = ttk.Label(root, text="Raw Weights Directory:", anchor="e")
raw_weights_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
raw_weights_entry = ttk.Entry(root, width=50)
raw_weights_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
raw_weights_entry.insert(0, config_values.get('raw_weights_dir', ''))
raw_weights_button = ttk.Button(root, text="Browse", command=browse_weights, style='Nice.TButton')
raw_weights_button.grid(row=1, column=2, padx=5, pady=5, sticky="w")

# Bits per Head Checkboxes
bits_per_head_label = ttk.Label(root, text="Bits per Head:", anchor="e")
bits_per_head_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")

bits_per_head_vars = []
bits_per_head_checkboxes = []
bits_per_head_options = ['6', '8']  # Add more options as needed

bits_per_head_frame = ttk.Frame(root)
bits_per_head_frame.grid(row=2, column=1, columnspan=2, padx=10, pady=5, sticky="w")

for i, bits_per_head in enumerate(bits_per_head_options):
    bits_per_head_var = tk.StringVar(value="")
    bits_per_head_checkbox = ttk.Checkbutton(bits_per_head_frame, text=bits_per_head, variable=bits_per_head_var, onvalue=bits_per_head, offvalue="")
    bits_per_head_checkbox.grid(row=0, column=i, padx=5, pady=5)
    bits_per_head_vars.append(bits_per_head_var)
    bits_per_head_checkboxes.append(bits_per_head_checkbox)

# Load previously selected Bits per Head values
if 'bits_per_head' in config_values:
    selected_bits_per_head_values = config_values['bits_per_head'].split(',')
    for bits_per_head_var in bits_per_head_vars:
        if bits_per_head_var.get() in selected_bits_per_head_values:
            bits_per_head_var.set(bits_per_head_var.get())

# BPW Values Checkboxes
bpw_label = ttk.Label(root, text="BPW Values:", anchor="e")
bpw_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

bpw_vars = []
bpw_checkboxes = []
bpw_options = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]  # Add more options as needed

bpw_frame = ttk.Frame(root)
bpw_frame.grid(row=3, column=1, columnspan=2, padx=10, pady=5, sticky="w")

for i, bpw in enumerate(bpw_options):
    bpw_var = tk.StringVar(value="")
    bpw_checkbox = ttk.Checkbutton(bpw_frame, text=str(bpw), variable=bpw_var, onvalue=str(bpw), offvalue="")
    bpw_checkbox.grid(row=i // 7, column=i % 7, padx=5, pady=5)
    bpw_vars.append(bpw_var)
    bpw_checkboxes.append(bpw_checkbox)

# Load previously selected BPW values
if 'bpw_values' in config_values:
    selected_bpw_values = config_values['bpw_values'].split(',')
    for bpw_var in bpw_vars:
        if bpw_var.get() in selected_bpw_values:
            bpw_var.set(bpw_var.get())

# Custom BPW Values Entry
custom_bpw_label = ttk.Label(root, text="Custom BPW Values (comma-separated):", anchor="e")
custom_bpw_label.grid(row=5, column=0, padx=10, pady=5, sticky="e")
custom_bpw_entry = ttk.Entry(root, width=50)
custom_bpw_entry.grid(row=5, column=1, columnspan=2, padx=10, pady=5, sticky="w")

# Load previously entered custom BPW values
if 'bpw_values' in config_values:
    custom_bpw_values = [bpw for bpw in selected_bpw_values if bpw not in map(str, bpw_options)]
    custom_bpw_entry.insert(0, ','.join(custom_bpw_values))

# Virtual Environment Path
venv_path_label = ttk.Label(root, text="Virtual Environment Path:", anchor="e")
venv_path_label.grid(row=6, column=0, padx=10, pady=5, sticky="e")
venv_path_entry = ttk.Entry(root, width=50)
venv_path_entry.grid(row=6, column=1, padx=10, pady=5, sticky="w")
venv_path_entry.insert(0, config_values.get('venv_path', ''))
venv_path_button = ttk.Button(root, text="Browse", command=browse_venv, style='Nice.TButton')
venv_path_button.grid(row=6, column=2, padx=5, pady=5, sticky="w")

# CUDA Device Entry
cuda_device_label = ttk.Label(root, text="CUDA Device Number:", anchor="e")
cuda_device_label.grid(row=7, column=0, padx=10, pady=5, sticky="e")
cuda_device_entry = ttk.Entry(root, width=50)
cuda_device_entry.grid(row=7, column=1, columnspan=2, padx=10, pady=5, sticky="w")
cuda_device_entry.insert(0, config_values.get('cuda_device', '0'))

# Exllamav2 Directory
exllamav2_dir_label = ttk.Label(root, text="Exllamav2 Directory:", anchor="e")
exllamav2_dir_label.grid(row=8, column=0, padx=10, pady=5, sticky="e")
exllamav2_dir_entry = ttk.Entry(root, width=50)
exllamav2_dir_entry.grid(row=8, column=1, padx=10, pady=5, sticky="w")
exllamav2_dir_entry.insert(0, config_values.get('exllamav2_dir', ''))
exllamav2_dir_button = ttk.Button(root, text="Browse", command=browse_exllamav2, style='Nice.TButton')
exllamav2_dir_button.grid(row=8, column=2, padx=5, pady=5, sticky="w")

# Measurement.json Path
measurement_path_label = ttk.Label(root, text="IF you already have measurement.json (optional):", anchor="e")
measurement_path_label.grid(row=9, column=0, padx=10, pady=5, sticky="e")
measurement_path_entry = ttk.Entry(root, width=50)
measurement_path_entry.grid(row=9, column=1, padx=10, pady=5, sticky="w")
measurement_path_entry.insert(0, config_values.get('measurement_path', ''))
measurement_path_button = ttk.Button(root, text="Browse", command=browse_measurement, style='Nice.TButton')
measurement_path_button.grid(row=9, column=2, padx=5, pady=5, sticky="w")

# Author Name
author_name_label = ttk.Label(root, text="Author Name (optional):", anchor="e")
author_name_label.grid(row=10, column=0, padx=10, pady=5, sticky="e")
author_name_entry = ttk.Entry(root, width=50)
author_name_entry.grid(row=10, column=1, columnspan=2, padx=10, pady=5, sticky="w")
author_name_entry.insert(0, config_values.get('author_name', ''))

# Progress Bar
progress_bar = ttk.Progressbar(root, mode='indeterminate', length=200)
progress_bar.grid(row=12, column=1, columnspan=2, padx=10, pady=10)
progress_bar.grid_forget()  # Hide the progress bar initially

# Run Button
run_button = ttk.Button(root, text="Run", command=start_run_commands, style='Nice.TButton')
run_button.grid(row=11, column=1, columnspan=2, padx=10, pady=20, sticky="w")

# Start the GUI event loop
root.mainloop()
