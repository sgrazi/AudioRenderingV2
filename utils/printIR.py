import matplotlib.pyplot as plt

def plot_data_from_file(filename):
    # Lists to store x and y data
    x_data = []
    y_data = []
    
    # Read data from file
    with open(filename, 'r') as file:
        for line in file:
            # Convert line to float
            value = float(line.strip())
            
            # Append value to y_data
            y_data.append(value)
            
            # Append corresponding index to x_data
            x_data.append(len(y_data))  # Use index as x-value
    
    # Plot the data
    plt.plot(x_data, y_data)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Data from File')
    plt.grid(True)
    plt.show()

# Example usage:
filename = 'arrayTest.txt'
plot_data_from_file(filename)
