import graphviz
dot = graphviz.Digraph("Decision Tree credit risk")

dot.node('Income', shape = 'diamond')
dot.node('Debt', shape = 'diamond')
# dot.node('Gender', shape = 'diamond')
dot.node('Married', shape = 'diamond')

dot.node('B', label='Low', shape = 'box')
dot.node('C', label='Low', shape = 'box')
dot.node('E', label='High', shape = 'box')
dot.node('F', label='High', shape = 'box')
dot.node('G', label='Low', shape = 'box')
dot.node('H', label='High', shape = 'box')
# dot.node('J', label='Low', shape = 'box')
# dot.node('I', label='High', shape = 'diamond')


dot.edge('Income', 'B',  'Medium' , fontsize='12')
dot.edge('Income', 'C', 'High', fontsize='12')
dot.edge('Income', 'Married', 'Low', fontsize='12')
dot.edge('Married', 'F', 'Yes', fontsize='12')
dot.edge('Married', 'Debt', 'No', fontsize='12')
dot.edge('Debt', 'E', 'High', fontsize='12')
dot.edge('Debt', 'G', 'Low',  fontsize='12')
dot.edge('Debt', 'H', 'Medium', fontsize='12')

# dot.edge('Gender', 'H', 'Male', fontsize='12')
# dot.edge('Gender', 'J', 'Female', fontsize='12')


print(dot.source)

dot.render('graphviz-output/credit_risk_latest.gv', view=True) 
dot.render(format='svg', directory='graphviz-output').replace('\\', '/')