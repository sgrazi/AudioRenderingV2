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
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of numbers from file')
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