import pandas as pd
from scipy.stats import entropy
from graphviz import Digraph

class DesicionTreeNode:
	def __init__(self, information = 0, value = '', next_branch_variable = '', samples = 0):
		self.information = information
		self.value = value
		self.next_branch_variable = next_branch_variable
		self.children = []
		self.samples = samples
	def __repr__(self):
		return f'Value= {self.value}\nSamples= {self.samples}\nInfo= {self.information}\nNext Branch= {self.next_branch_variable}\nChildren= {len(self.children)}'', '

def information(data, classAttribute):
	return entropy(data[classAttribute].value_counts(), base=2)

def createDecisionTree(data, classAttribute, root=None):
	root_info = information(data, classAttribute)
	data_count = data.shape[0]
	
	if root_info == 0 or data_count == 0 or data.shape[1] == 1:
		return

	info = {}
	#for each attribute in data
	for attribute in data.drop([classAttribute], axis=1).keys():
		info[attribute] = 0
		#for each unique value in that variable
		for unique_value in data[attribute].unique():
			data_i = data[data[attribute] == unique_value]
			info_i = information(data_i, classAttribute)
			p_i = data_i.shape[0] / data_count
			info[attribute] += p_i * info_i
	#choose the highest info_nextLayer and choose that variable
	next_branch_variable = min(info, key=info.get)
	
	if root == None:
		root = DesicionTreeNode(information = root_info, value = 'root', next_branch_variable = next_branch_variable, samples = data_count)
	else:
		root.information = root_info
		root.next_branch_variable = next_branch_variable

	for unique_value in data[next_branch_variable].unique():
		newData = data[data[next_branch_variable] == unique_value].drop(columns=[next_branch_variable],axis=1)
		childNode = DesicionTreeNode(value = unique_value, samples = newData.shape[0])
		createDecisionTree(newData, classAttribute, childNode)
		root.children.append(childNode)
	return root

def createGraph(root, graph = None):
	if graph == None:
		graph = Digraph('G', filename='dt.gv', node_attr={'shape': 'record'})
	for child in root.children:
		graph.edge(str(root), str(child))
		createGraph(child, graph)
	return graph

def main():
	data = []
	data += 30 * [['sales', 'senior', '31-35', '46-50K']]
	data += 40 * [['sales', 'junior', '26-30', '26-30K']]
	data += 40 * [['sales', 'junior', '31-35', '31-35K']]
	data += 20 * [['systems', 'junior', '21-25', '46-50K']]
	data += 5 * [['systems', 'senior', '31-35', '66-70K']]
	data += 3 * [['systems', 'junior', '26-30', '46-50K']]
	data += 3 * [['systems', 'senior', '41-45', '66-70K']]
	data += 10 * [['marketing', 'senior', '36-40', '46-50K']]
	data += 4 * [['marketing', 'junior', '31-35', '41-45K']]
	data += 4 * [['secretary', 'senior', '46-50', '36-40K']]
	data += 6 * [['secretary', 'junior', '26-30', '26-30K']]
	labels = ['department', 'status', 'age', 'salary']
	df = pd.DataFrame(data, columns=labels)

	data2 = []
	data2 += [['Sunny', 'Hot', 'High', 'False', 'No']]
	data2 += [['Sunny', 'Hot', 'High', 'True', 'No']]
	data2 += [['Overcast', 'Hot', 'High', 'False', 'Yes']]
	data2 += [['Rainy', 'Mild', 'High', 'False', 'Yes']]
	data2 += [['Rainy', 'Cool', 'Normal', 'False', 'Yes']]
	data2 += [['Rainy', 'Cool', 'Normal', 'True', 'No']]
	data2 += [['Overcast', 'Cool', 'Normal', 'True', 'Yes']]
	data2 += [['Sunny', 'Mild', 'High', 'False', 'No']]
	data2 += [['Sunny', 'Cool', 'Normal', 'False', 'Yes']]
	data2 += [['Rainy', 'Mild', 'Normal', 'False', 'Yes']]
	data2 += [['Sunny', 'Mild', 'Normal', 'True', 'Yes']]
	data2 += [['Overcast', 'Mild', 'High', 'True', 'Yes']]
	data2 += [['Overcast', 'Hot', 'Normal', 'False', 'Yes']]
	data2 += [['Rainy', 'Mild', 'High', 'True', 'No']]
	labels2 = ['Outlook', 'Temperature', 'Humidity', 'Windy', 'Play']
	df2 = pd.DataFrame(data2, columns=labels2)

	createGraph(createDecisionTree(data = df, classAttribute='status')).view()

if __name__ == '__main__':
	main()