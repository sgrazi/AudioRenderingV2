import matplotlib.pyplot as plt
import sys, getopt

def plot_numbers_from_file(filepath):
    # Step 1: Read the file
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Step 2: Parse the numbers
    numbers = [float(line.strip()) for line in lines]

    # Step 3: Plot the numbers
    plt.plot(numbers)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('IR')

    # 16k sample rate
    # Calculate the number of data points and the x-tick positions for 250ms intervals
    num_data_points = len(numbers)
    num_intervals = num_data_points // 4000
    x_tick_positions = [250 * i for i in range(num_intervals + 1)]

    # Set x-tick labels and positions
    plt.xticks(x_tick_positions, [f'{int(x)}ms' for x in x_tick_positions])
    
    plt.show()

def main(argv):
   filepath = ''
   outputfile = ''
   opts, args = getopt.getopt(argv,"hf:o")
   for opt, arg in opts:
      if opt == '-h':
         print ('printIR.py -f <filepath> -o <outputfile>')
         sys.exit()
      elif opt == "-f":
         filepath = arg
      elif opt == "-o":
         outputfile = arg
   plot_numbers_from_file(filepath)


if __name__ == "__main__": 
  main(sys.argv[1:])