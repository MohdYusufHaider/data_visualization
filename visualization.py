#Q1> Create a scatter plot using Matplotlib to visualize the relationship between two arrays, x and y for the given
#data.
#x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
#y = [2, 4, 5, 7, 6, 8, 9, 10, 12, 13]
import matplotlib.pyplot as plt
import numpy as np
x= ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = ([2, 4, 5, 7, 6, 8, 9, 10, 12, 13])
plt.scatter(x,y)
plt.show()

#Q2> Generate a line plot to visualize the trend of values for the given data.
#data = np.array([3, 7, 9, 15, 22, 29, 35])





data = np.array([3, 7, 9, 15, 22, 29, 35])
x = np.arange(len(data))


plt.plot(x, data, marker='o', linestyle='-', color='b', label='Trend Line')


plt.title('Trend of Data Values')
plt.xlabel('Index')
plt.ylabel('Value')


plt.legend()


plt.grid(True)


plt.show()
plt.plot(x, data, marker='o', linestyle='-', color='b', label='Trend Line')


plt.title('Trend of Data Values')
plt.xlabel('Index')
plt.ylabel('Value')


plt.legend()


plt.grid(True)


plt.show()

#Q3>Display a bar chart to represent the frequency of each item in the given array categories?
categories = ['A', 'B', 'C', 'D', 'E']
values = [25, 40, 30, 35, 20]
plt.bar(categories, values, color='skyblue')


plt.title('Frequency of Categories')
plt.xlabel('Categories')
plt.ylabel('Frequency')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#Q4>Create a histogram to visualize the distribution of values in the array data?
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, color='purple', edgecolor='black', alpha=0.7)


plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')


plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#Q5> Show a pie chart to represent the percentage distribution of different sections in the array `sections
sections = ['Section A', 'Section B', 'Section C', 'Section D']
sizes = [25, 30, 15, 30]
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=sections, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})


plt.title('Percentage Distribution of Sections')
plt.show()



       #########SEA BORN #######
       
#Q1>Create a scatter plot to visualize the relationship between two variables, by generating a synthetic
#dataset
import seaborn as sns
np.random.seed(100)  # Setting seed for reproducibility

# Generating random data for two variables (x and y)
x = np.random.uniform(0, 50, 100)  # 100 random values between 0 and 50
y = 2 * x + np.random.normal(0, 10, 100)  # Linear relationship with some random noise

# Plotting the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='teal', marker='o', edgecolor='black', alpha=0.7, label='Data Points')


plt.title('Scatter Plot of Synthetic Data')
plt.xlabel('Variable X')
plt.ylabel('Variable Y')


plt.grid(True, linestyle='--', alpha=0.5)

plt.legend()


plt.show()

#Q2>> Generate a dataset of random numbers. Visualize the distribution of a numerical variable.
np.random.seed(42)  # For reproducibility
data = np.random.randn(1000)  # Random numbers from standard normal distribution (mean=0, std=1)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, color='dodgerblue', edgecolor='black', alpha=0.7)


plt.title('Distribution of a Numerical Variable')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()

#Q3> Create a dataset representing categories and their corresponding values. Compare different categories
#based on numerical values
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Creating a dataset with categories and corresponding numerical values
data = {
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value': [25, 40, 30, 35, 20]
}


df = pd.DataFrame(data)


plt.figure(figsize=(8, 6))
sns.barplot(x='Category', y='Value', data=df, palette='viridis')


plt.title('Comparison of Different Categories')
plt.xlabel('Category')
plt.ylabel('Value')


plt.show()

#Q4>Generate a dataset with categories and numerical values. Visualize the distribution of a numerical
#variable across different categories



np.random.seed(42)  
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.rand(100) * 50 
category_labels = np.random.choice(categories, size=100)  

# Creating a DataFrame
df = pd.DataFrame({
    'Category': category_labels,
    'Value': values
})

# Visualizing the distribution of numerical values across different categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Value', data=df, palette='Set2')


plt.title('Distribution of Numerical Values Across Categories')
plt.xlabel('Category')
plt.ylabel('Value')

# Displaying the plot
plt.show()

#Q5>Generate a synthetic dataset with correlated features. Visualize the correlation matrix of a dataset using a
#heatmap



np.random.seed(42)

x = np.random.rand(100)


y = 0.8 * x + np.random.normal(0, 0.1, 100)  # y is correlated with x
z = 0.5 * x + 0.2 * y + np.random.normal(0, 0.1, 100)  # z is correlated with x and y

# Create a DataFrame
df = pd.DataFrame({
    'Feature 1': x,
    'Feature 2': y,
    'Feature 3': z
})

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Visualizing the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Adding a title
plt.title('Correlation Matrix of the Synthetic Dataset')

# Displaying the heatmap
plt.show()

   #######PLOTLY #########
   
   
#Q1> Using the given dataset, to generate a 3D scatter plot to visualize the distribution of data points in a threedimensional space

np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}
df = pd.DataFrame(data)


from mpl_toolkits.mplot3d import Axes3D


np.random.seed(30)
data = {
    'X': np.random.uniform(-10, 10, 300),
    'Y': np.random.uniform(-10, 10, 300),
    'Z': np.random.uniform(-10, 10, 300)
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Creating a 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting the data points
ax.scatter(df['X'], df['Y'], df['Z'], c='b', marker='o', alpha=0.6)

# Adding labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Scatter Plot of Data Points')

# Displaying the plot
plt.show()


#Q2> Using the Student Grades, create a violin plot to display the distribution of scores across different grade
#categories
np.random.seed(15)
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}
df = pd.DataFrame(data)
                 

np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}
df = pd.DataFrame(data)

np.random.seed(15)
data = {
    'Grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 200),
    'Score': np.random.randint(50, 100, 200)
}

df_grades = pd.DataFrame(data)

# Creating the violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Grade', y='Score', data=df_grades, palette='Set2')

# Adding titles and labels
plt.title('Distribution of Scores Across Different Grades')
plt.xlabel('Grade')
plt.ylabel('Score')

# Displaying the plot
plt.show()



###2


np.random.seed(20)
data = {
    'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May'], 100),
    'Day': np.random.choice(range(1, 31), 100),
    'Sales': np.random.randint(1000, 5000, 100)
}

df_sales = pd.DataFrame(data)

# Pivoting the data for heatmap visualization
sales_pivot = df_sales.pivot_table(index='Day', columns='Month', values='Sales', aggfunc='mean')

# Creating the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(sales_pivot, cmap='YlGnBu', annot=True, fmt='.0f', linewidths=0.5)

# Adding titles and labels
plt.title('Variation in Sales Across Different Months and Days')
plt.xlabel('Month')
plt.ylabel('Day')

# Displaying the heatmap
plt.show()


#Q4>Using the given x and y data, generate a 3D surface plot to visualize the function 
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))
data = {
    'X': x.flatten(),
    'Y': y.flatten(),
    'Z': z.flatten()
}
df = pd.DataFrame(data)



# Generating x, y data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)  # Create meshgrid from x and y

# Calculating z = sin(sqrt(x^2 + y^2))
z = np.sin(np.sqrt(x**2 + y**2))

# Creating the DataFrame
data = {
    'X': x.flatten(),
    'Y': y.flatten(),
    'Z': z.flatten()
}
df = pd.DataFrame(data)

# Creating a 3D surface plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plotting the surface
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Adding labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Surface Plot of z = sin(sqrt(x^2 + y^2))')

# Displaying the plot
plt.show()


#Q5> Using the given dataset, create a bubble chart to represent each country's population (y-axis), GDP (xaxis), and bubble size proportional to the population
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK',
'Germany', 'France'],
    'Population':
np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000,
5)
}
df = pd.DataFrame(data)





# Generating the dataset
np.random.seed(25)
data = {
    'Country': ['USA', 'Canada', 'UK', 'Germany', 'France'],
    'Population': np.random.randint(100, 1000, 5),
    'GDP': np.random.randint(500, 2000, 5)
}

df = pd.DataFrame(data)

# Creating the bubble chart
plt.figure(figsize=(10, 6))

# Bubble chart: x = GDP, y = Population, size = Population
plt.scatter(df['GDP'], df['Population'], s=df['Population']*10, c=df['Population'], cmap='viridis', alpha=0.6, edgecolors="w", linewidth=1.5)

# Adding titles and labels
plt.title('Bubble Chart: Population vs GDP', fontsize=16)
plt.xlabel('GDP (in billions)', fontsize=12)
plt.ylabel('Population (in millions)', fontsize=12)

# Annotating the points with country names
for i in range(len(df)):
    plt.text(df['GDP'][i] + 10, df['Population'][i], df['Country'][i], fontsize=12)

# Display the plot
plt.show()



#Q1>Create a Bokeh plot displaying a sine wave. Set x-values from 0 to 10 and y-values as the sine of x

from bokeh.plotting import figure, show


# Generate x-values from 0 to 10
x = np.linspace(0, 10, 100)

# Calculate y-values as the sine of x
y = np.sin(x)

# Create a Bokeh plot
p = figure(title="Sine Wave", x_axis_label='x', y_axis_label='sin(x)', width=800, height=400)

# Add a line renderer for the sine wave
p.line(x, y, legend_label="sin(x)", line_width=2, color="blue")

# Show the plot
show(p)
#Q3>Create a Bokeh scatter plot using randomly generated x and y values. Use different sizes and colors for the
#markers based on the 'sizes' and 'colors' columns



# Generate random data for x, y, sizes, and colors
np.random.seed(42)
x = np.random.random(100) * 10  
y = np.random.random(100) * 10 
sizes = np.random.randint(10, 50, 100)  
colors = np.random.choice(['red', 'green', 'blue', 'orange', 'purple'], 100)  # Random colors

data = pd.DataFrame({
    'x': x,
    'y': y,
    'sizes': sizes,
    'colors': colors
})

p = figure(title="Scatter Plot with Random Sizes and Colors", 
           x_axis_label='X', 
           y_axis_label='Y', 
           width=800, 
           height=400)

# Add scatter plot with size and color based on the 'sizes' and 'colors' columns
p.scatter('x', 'y', source=data, size='sizes', color='colors', legend_field='colors', fill_alpha=0.6)

# Adding legend and title
p.legend.title = 'Colors'
p.legend.location = "top_left"

# Show the plot
show(p)

#Q5>Create a Bokeh heatmap using the provided dataset.
data_heatmap = np.random.rand(10, 10)
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
xx, yy = np.meshgrid(x, y)



# Provided data
data_heatmap = np.random.rand(10, 10)  
x = np.linspace(0, 1, 10)  
y = np.linspace(0, 1, 10)  
xx, yy = np.meshgrid(x, y)

# Create a Bokeh plot
p = figure(title="Heatmap Example", x_axis_label='X', y_axis_label='Y', width=600, height=600)

# Add image renderer for the heatmap data
p.image(image=[data_heatmap], x=0, y=0, dw=1, dh=1, color_mapper="Viridis256")

# Customize plot properties
p.x_range.range_padding = 0
p.y_range.range_padding = 0
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

# Show the plot
show(p)

#Q4> Create a Bokeh histogram to visualize the distribution of the given data.
data_hist = np.random.randn(1000)
hist, edges = np.histogram(data_hist, bins=30)


from bokeh.plotting import figure, show
import numpy as np

# Generate random data
data_hist = np.random.randn(1000)  # 1000 random values from a normal distribution
hist, edges = np.histogram(data_hist, bins=30)  # Compute histogram with 30 bins

# Create a Bokeh plot
p = figure(title="Histogram of Random Data", 
           x_axis_label='Value', 
           y_axis_label='Frequency', 
           width=800, 
           height=400)

# Add histogram bars to the plot
p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], 
       fill_color="skyblue", line_color="white", alpha=0.7)

# Show the plot
show(p)
