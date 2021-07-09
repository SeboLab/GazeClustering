# Import pandas package
import pandas as pd
 
# Function to add
def add(a, b, c):
    return a + b + c
 
def main():
     
    # create a dictionary with
    # three fields each
    data = {
            'A':[1, 2, 3],
            'B':[4, 5, 6],
            'C':[7, 8, 9] }
     
    # Convert the dictionary into DataFrame
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)
     
    df['add'] = df.apply(lambda row : add(row['A'],
                     row['B'], row['C']), axis = 1)
  
    print('\nAfter Applying Function: ')
    # printing the new dataframe
    print(df)
  
if __name__ == '__main__':
    main()