# How to plot the IR reponse from a file

## Setup

1. **Create and activate a virtual environment**:

   - **Create Virtual Environment**:

     ```bash
     python -m venv .venv
     ```

   - **Activate the Virtual Environment**:

     - **On Windows**:

       ```bash
       .\.venv\Scripts\activate
       ```

     - **On macOS and Linux**:

       ```bash
       source .venv/bin/activate
       ```

2. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Program

After setting up the environment, you can run the program with:

```bash
python main.py -f <path_to_file> -o <output_file>
```

Replace `<path_to_file>` with the path to your file containing numbers and use `<output_file>` if you want to export the graph to a png.

## Deactivating the Virtual Environment

Once you're done, you can deactivate the virtual environment with:

```bash
   deactivate
   ```
