# Standardization

Hello everyone, so today we will explore the concept of Feature Scaling in Machine Learning.

What is Standardization in Machine Learning...?

In Machine Learning we train our data to predict or classify things in such a manner that isn't hardcoded in the machine.

So for the first we have the Dataset or the input data to be pre-processed and manipulated for our desired outcomes. Any ML Model to be built follows the following procedure:

Collect Data
Perform Data Munging/Cleaning (Feature Scaling)
Pre-Process Data
Apply Visualizations
And so on...



Feature Scaling is divided into:

Standardization
Normalization



Our interest of the following topic (Standardization) lies in the first 3 steps, and we will dive into that right now.

1] Considering STEP 1, we have collected our data:

Our data can be in various formats i.e., numbers (integers) & words (strings), for now we'll consider only the numbers in our Dataset.

Assume our dataset has random numeric values in the range 1 to 95,000. Obviously in random order though. Just for our understanding consider a small Dataset of barely 10 values with numbers in the given range and randomized order.

1)  99
2)  789
3)  1
4)  541
5)  5
6)  6589
7)  94142
8)  7
9)  50826
10) 35464

If we just look at these values, their range is so high, that while training the model with 10,000 such values will take lot of time. So we have a problem .



2] Going to Step 2:

But we have a solution for the same and a very prominent one! Standardization helps us solve this by 

Down Scaling the Values to a scale common to all, usually in the range -1 to +1.
And keeping the Range between the values intact.

So, how do we do that? we'll there's a mathematical formula for the same i.e., Z-Score = (Current_value - Mean) / Standard Deviation.



Using this formula we are replacing all the input values by the Z-Score for each and every value.

Hence we get values ranging from -1 to +1, keeping the range intact.



Standardization performs the following:

Converts the Mean (μ) to 0
Converts to S.D. (σ) to 1

It's pretty obvious for Mean =0 and S.D = 1 as all the values will have such less difference and each value will nearly be equal 0, hence Mean = 0 and S.D. = 1.



NOTE : (Just for Better Understanding)

For Mean

When we Subtract a value Smaller than the Mean we get (-ve) Output
When we Subtract a value Larger than the Mean we get (+ve) Output

Hence, when we get (-ve) & (+ve) Values for Subtraction of Value with Mean, while Summation of all these values,
We get the Final Mean as 0.

And when we get the Mean as 0, it means that most or nearly all values are equal to to highly close to 0 and have very low variance among them.

Therefore, the S.D also becomes 1 (as good as no difference).



Let's look at the Execution/ Implementation now



Here we are doing the Following:

Calculating the Z-Score
Comparing the Original Values ad Standardized Values
Comparing the Range of both using Scatter Plots

Python3

# We are just using 10 values for our Dataset

# Here, dataset_0 will be constant as it's range is just 1 - 10
# But, dataset_1 will be scaled down as it's range is 1 - 95,000
import matplotlib
import matplotlib.pyplot as plt
global dataset_0, dataset_1
dataset_0 = [10, 5, 6, 1, 3, 7, 9, 4, 8, 2]
dataset_1 = [1, 99, 789, 5, 6859, 541, 94142, 7, 50826, 35464]

n = len(dataset_1)
mean_ans = 0
ans = 0
j = 0

for i in dataset_1:
    j = j + i
    k = i*i
    ans = ans + k

print('n : ', n)
print("Summation (X)   : ", j)
print("Summation (X^2) : ", ans)

part_1 = ans/n
part_2 = mean_ans*mean_ans
standard_deviation = part_1 - part_2
print("\nStandard Deviation : ", standard_deviation)

mean = j/n
print("Mean               : ", mean)
print("\n\n\n")


# Calculating the Z-Score for each Value of dataset_1
final_z_score = []
print("Calculating Z-Score of Each Value in dataset_1")
for i in dataset_1:
    z_score = (i-mean)/standard_deviation
    final_z_score.append("{:.20f}".format(z_score))
    print(i, "-", mean, "/", standard_deviation)
    print("Z - Score(", i, ") : {:.20f}".format(z_score))
    print("\n")


# Comparing the Values of Original Dataset and Saled Down Dataset
# print("\n")
print("\nOriginal DataSet   |               Z-Score ")
print()
for i in range(len(dataset_1)):
    print("    ", dataset_1[i], "            |     ", final_z_score[i])


# Now we will compare and see the graph of the Original Values and the Standardized Values

print("\n\n")
# Here We are checking the Graph of the Original Values
plt.scatter(dataset_0, dataset_1, label="stars",
            color="blue", marker="*", s=40)

plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('Original Values')
plt.legend()

plt.show()


print("\n\n")
# Here we are checking the Graph of the Standardized Values
plt.scatter(dataset_0, final_z_score, label="stars",
            color="blue", marker="*", s=30)

plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('Original Values')
plt.legend()

plt.show()




Calculating the Z-Score now for each and every value of dataset_1

Python3

final_z_score = []

for i in dataset_1:
    z_score = (i-mean)/standard_deviation
    final_z_score.append("{:.20f}".format(z_score))
    print(i, "-", mean, "/", standard_deviation)
    print("Z - Score(", i, ") : {:.20f}".format(z_score))
    print("\n")




Comparing the Original Value and the Scaled Down Values

Python3

print("  Original DataSet   |               Z-Score ")
print()
for i in range(len(dataset_1)):
    print("         ", dataset_1[i], "         |     ", final_z_score[i])




Comparing the Range of Values using Graphs



1) Graph of the Original Values

Python3

#Now we will compare and see the graph of the Original Values and the Standardized Values
import matplotlib 
import matplotlib.pyplot as plt

#Here We are checking the Graph of the Original Values
plt.scatter(dataset_0, dataset_1, label= "stars", color= "blue", marker= "*", s=40)
    
plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('Original Values')
plt.legend()

plt.show()



2) Graph of the Standardized Values

Python3

import matplotlib 
import matplotlib.pyplot as plt

plt.scatter(dataset_0, final_z_score, label= "stars", color= "blue", marker= "*", s=30)
  
plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.title('Original Values')
plt.legend()
  
plt.show()




Hence we have Reviewed, Understood the Concept and Implemented as well the Concept of Standardization in Machine Learning
