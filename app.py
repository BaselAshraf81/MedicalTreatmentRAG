import streamlit as st
import subprocess

# Function to run shell commands
def run_command(command):
    """Run a shell command and return the output."""
    try:
        # Using subprocess to run the command
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

# Streamlit UI
st.title("Shell Command Runner")

# Command to uninstall opentelemetry-proto
uninstall_command = "pip uninstall -y opentelemetry-proto"
if st.button("Uninstall opentelemetry-proto"):
    output = run_command(uninstall_command)
    st.text(output)

# Command to install opentelemetry-proto
install_command = "pip install opentelemetry-proto"
if st.button("Install opentelemetry-proto"):
    output = run_command(install_command)
    st.text(output)

# Displaying the output of the commands
if st.button("Check Installed Version"):
    version_command = "pip show opentelemetry-proto"
    output = run_command(version_command)
    st.text(output)
