import matplotlib.pyplot as plt
import glob
import re

def simpleRegexPlot():
  desc = """
  Ingrese un expresión regular para indicar cuales archivos necesitan ser graficados:
      i.e: *.txt para graficar todos los archivos de texto dentro de la carpeta inputs
  """
  file_regex = input(desc)
  file_names = glob.glob(f"inputs/{file_regex}")
  for file_name in file_names:
    x_data = []
    y_data = []
    first_index = 0
    last_index = 0
    with open(file_name, 'r') as file:
        for (index, line) in enumerate(file):
            # Convert line to float
            value = float(line.strip())
            
            # Append value to y_data
            if first_index == 0 and value != 0:
              first_index = index
            if (value != 0):
              last_index = index
            y_data.append(value)
            x_data.append(index)
    
    # round last_index to the next 100
    module = last_index % 100
    last_index += 100 - module
    
    # round first_index to the previous 100
    module = first_index % 100
    first_index -= 100 - module
    
    y_data = y_data[first_index:last_index]
    x_data = x_data[first_index:last_index]
    
    defaultDpi = 300
    plt.figure(figsize=(1920/defaultDpi, 1080/defaultDpi), dpi=defaultDpi)
    plt.plot(x_data, y_data)
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    print(file_name)
    plot_title = file_name.split("\\")[1].split(".txt")[0]
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(f"outputs/{plot_title}.png")
    plt.close()
  
def regexPlotLR():
  desc = """
  Ingrese un expresión regular para indicar cuales archivos necesitan ser graficados:
      i.e: *.txt para graficar todos los archivos de texto dentro de la carpeta inputs
  """
  file_regex = input(desc)
  file_names = glob.glob(f"inputs/{file_regex}")
  LR_file_names = set()
  
  for file_name in file_names:
    value = re.search(r".*(?=_[LR].txt)", file_name)
    LR_file_names.add(value[0])
  
  for file_name in LR_file_names:
    x_data_L = []
    y_data_L = []
    x_data_R = []
    y_data_R = []

    first_index = 0
    last_index = 0
    file_name_L = f"{file_name}_L.txt"
    file_name_R = f"{file_name}_R.txt"
    with open(file_name_L, 'r') as file:
        for (index, line) in enumerate(file):
            value = float(line.strip())
            if first_index == 0 and value != 0:
              first_index = index
            if (value != 0):
              last_index = index
            x_data_L.append(index)
            y_data_L.append(value)
    
    with open(file_name_R, 'r') as file:
        for (index, line) in enumerate(file):
            value = float(line.strip())
            if first_index == 0 and value != 0:
              first_index = index
            if last_index < index and (value != 0):
              last_index = index
            x_data_R.append(index)
            y_data_R.append(value)
    
    # round last_index to the next 100
    module = last_index % 100
    last_index += 100 - module
    
    # round first_index to the previous 100
    module = first_index % 100
    first_index -= 100 - module
    
    x_data_L = x_data_L[first_index:last_index]
    x_data_R = x_data_R[first_index:last_index]
    
    y_data_L = y_data_L[first_index:last_index]
    y_data_R = y_data_R[first_index:last_index]
    
    defaultDpi = 300
    plt.figure(figsize=(1920/defaultDpi, 1080/defaultDpi), dpi=defaultDpi)
    plt.plot(x_data_L, y_data_L, label='Left Channel')
    plt.plot(x_data_R, y_data_R, label='Right Channel')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    print(file_name)
    plot_title = file_name.split("\\")[1]
    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"outputs/{plot_title}_LR.png")
    plt.close()
  
def main():
  description= """
  ----------------------------------
  1. SimpleRegexPlot
  2. RegexPlotLR
  ----------------------------------
  """
  option = input(description)
  if option == "1":
    simpleRegexPlot()
  elif option == "2":
    regexPlotLR()
  else:
    print("Opción incorrecta")

if __name__ == "__main__":
  main()