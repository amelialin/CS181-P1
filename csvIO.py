#####################
# CS 181, Spring 2016
# Practical 1
# Steven, Amelia, Wouter
##################

# Import necessary libraries
import csv
import numpy as np
import matplotlib.pyplot as plt

#vars
arrays = []

#functions
def readFile(filename, arrays):    
    with open(filename, 'r') as csv_fh:
        # Read CSV file.
        reader = csv.reader(csv_fh)
        
        # Get first row of names and add an array for each column        
        row1 = next(reader)
        for col in row1:
            arrays.append([])
    
        # Loop over the file.        
        for row in reader:
            # Store the data.
            x = 0            
            for array in arrays:
                array.append(float(row[x]))
                x = x+1
    
def saveFile(filename, arrays):   
    nFile = open(filename, 'wb')
    csvFile = csv.writer(nFile)
    csvFile.writerows(arrays)
    nFile.close
 
#Unittest functions   
readFile('sample1.csv', arrays)
saveFile('outputTest.csv', np.transpose(arrays))