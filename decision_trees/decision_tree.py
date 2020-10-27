import numpy as np
from numpy.core.defchararray import replace
import pandas as pd
import matplotlib.pyplot as plt

def gini(x, col_name=None, thresh_val=None, debug=False):
    tgt_types = x['target'].unique()

    if(col_name == None):
        tot_no = x.shape[0]
        class_loss = np.zeros((tgt_types.size,1))
        
        for i in range(tgt_types.size):
            class_no = x.loc[x['target'] == tgt_types[i]].shape[0]
            class_p  = class_no / tot_no
            class_loss[i, 0] = class_p * (1 - class_p)

        return sum(class_loss)

    upper = x.loc[x[col_name] >  thresh_val]
    lower = x.loc[x[col_name] <= thresh_val]


    class_loss = np.zeros((tgt_types.size, 2))

    upper_tot = upper.shape[0]   
    lower_tot = lower.shape[0] 

    if(upper_tot == 0 or lower_tot == 0):
        return 1

    for i in range(tgt_types.size):
        upper_cls_no = upper.loc[upper['target'] == tgt_types[i]].shape[0]
        lower_cls_no = lower.loc[lower['target'] == tgt_types[i]].shape[0]
        
        upper_cls_p  = upper_cls_no / upper_tot
        lower_cls_p  = lower_cls_no / lower_tot

        class_loss[i, 0] = upper_cls_p * (1 - upper_cls_p)
        class_loss[i, 1] = lower_cls_p * (1 - lower_cls_p)

    gini = np.sum(class_loss, axis=0)

    impurity = (upper_tot*gini[0]/x.shape[0]) + (lower_tot*gini[1]/x.shape[0])
        
    return impurity


def split_data(x, col_name, thresh, drop=False):
    upper = x.loc[x[col_name] > thresh]
    lower = x.loc[x[col_name] <= thresh]

    if drop:
        return (upper.drop(col_name, axis='columns'), lower.drop(col_name, axis='columns'))
    else:
        return (upper, lower)


def visualize_split(x, col_name, thresh, tgt_classes, axis):

    data_target_sorted = {}
    for i in tgt_classes:
        data_target_sorted[i] = x.loc[x['target'] == i]

    for j in tgt_classes:
        data_target_j = data_target_sorted[j]
        axis.scatter(data_target_j[col_name], data_target_j['target'])
    
    axis.legend(tgt_classes)
    axis.axline((thresh,tgt_classes[0]), (thresh,tgt_classes[1]))

# Visualize tree using pyploy
def visualize_tree_working(x, root, levels, save=None):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    max_width = 2**(levels)

    tgt_classes = x['target'].unique()

    fig, ax = plt.subplots(levels+1, max_width)

    curr_nodes = [(root, int(max_width/2), 1, x)]
    all_nodes  = [(root, int(max_width/2), 1, x)]
    curr_mult  = max_width/2

    for i in ax:
        for j in i:
            j.axis('off')

    ax[0][int(max_width/2)].clear()
    ax[0][int(max_width/2)].axis('on')
    ax[0][int(max_width/2)].set_title(root.col)
    visualize_split(x, root.col, root.thresh, tgt_classes, ax[0][int(max_width/2)])

    for i in range(levels + 1):
        curr_level = i+1
        next_nodes = []
        
        for n, x_pos, y_pos, data in curr_nodes:
            upper_data, lower_data = n.split(data)
            upper_node, lower_node = n.upper, n.lower
            upper_x, lower_x = int(x_pos + curr_mult/2), int(x_pos - curr_mult/2)          
            upper_ax, lower_ax = ax[curr_level][upper_x], ax[curr_level][lower_x]

            upper_ax.clear()
            lower_ax.clear()
            upper_ax.axis('on')
            lower_ax.axis('on')

            if(curr_level != levels):
                ax[curr_level-1][upper_x].plot(100*np.ones(50), range(-50,0), 'b')
                ax[curr_level-1][upper_x].set_ylim(-50,50)
                ax[curr_level-1][upper_x].set_xlim(0,200)
                for j in range(x_pos+1, upper_x+1):
                    ax[curr_level-1][j].plot(range(100),np.zeros(100), 'b')
                
            ax[curr_level-1][lower_x].plot(np.zeros(50), range(-50,0), 'b')
            ax[curr_level-1][lower_x].set_ylim(-50,50)
            ax[curr_level-1][lower_x].set_xlim(-100,100)
            for j in range(lower_x, x_pos):
                ax[curr_level-1][j].plot(range(100),np.zeros(100), 'b')
                

            if(lower_node.leaf):
                lower_ax.fill((0,0.5,1),(0,0.5,0), color=colors[int(lower_node.state)])
            else:
                next_nodes.append((lower_node, lower_x, curr_level, lower_data))
                lower_ax.set_title(lower_node.col)
                visualize_split(lower_data, lower_node.col, lower_node.thresh, tgt_classes, lower_ax)

            if(upper_node.leaf):
                upper_ax.fill((0,0.5,1),(0,0.5,0), color=colors[int(upper_node.state)])
            else:
                next_nodes.append((upper_node, upper_x, curr_level, upper_data))
                upper_ax.set_title(upper_node.col)
                visualize_split(upper_data, upper_node.col, upper_node.thresh, tgt_classes, upper_ax)
                
        curr_mult = curr_mult/2
        curr_nodes = next_nodes

    fig.set_size_inches(2.5 * max_width, 2 * (levels+1))
    fig.show()
    
    if(save != None):
        print(" Saving output as %s"%save)
        fig.savefig(save)

def find_best_split(x, max_tries):
    inputs = x.drop('target', axis=1)
    outputs = x['target']
    
    min_impurity = 1
    min_col = ""
    min_thresh = -1

    for col_name in inputs.columns:

        values = x.sort_values(col_name)[col_name].unique()
        thresholds = np.asarray([(values[i]+values[i+1])/2 for i in range(values.shape[0]-1)])
        
        if(max_tries != None):
            if(len(thresholds) >= max_tries):
                thresholds = np.random.choice(thresholds, max_tries, replace=False)

        for i in thresholds:
            impurity = gini(x, col_name, i, False)

            if(impurity < min_impurity):
                min_impurity = impurity
                min_col      = col_name
                min_thresh   = i

    return (min_impurity, min_col, min_thresh)

class DecisionNode():
    def __init__(self, parent, level=None, name=None, debug=False):
        self.parent = parent
        
        if(level == None):
            if(parent == None):
                self.level = 0
            else:
                self.level = parent.level + 1
        else:
            self.level = level
        
        self.leaf   = False             # Is this node a leaf node?
        self.state  = None              # The output state (if a leaf node)
        self.col    = None              # Threshold column name
        self.thresh = 0                 # Threshold value
        
        self.upper  = None              # Upper child node
        self.lower  = None              # Lower child node

        self.debug  = debug             # If true, displays debug information

        if(name == None):               # The name is displayed if debug is turned on
            if(parent == None):
                self.name = "root_node"
            else:
                self.name = "level%d_node" % self.level
        else:
            self.name = name

        if(debug):
            print(" %15s : Initialised node with level : %2d" % (self.name, self.level))

    # Turn the node into a leaf node, with provided output
    def make_leaf(self, output):
        self.leaf  = True
        self.state = output

        if(self.debug):
            print(" %15s : Made into leaf node, with output '%d'" % (self.name, self.state))

    # Train with data (set column and threshold).
    # If the gini impurity has deteriorated or has not improved, then the node is made into a leaf node.
    # force_decision can be set to True to disable automatic conversion to leaf nodes
    def train(self, x, force_decision=False, max_tries=None):
        data_impurity = gini(x)
        impurity, self.col, self.thresh = find_best_split(x, max_tries)

        if(self.debug):
            print(" %15s : Trained. Impurity before : %.2f, Impurity : %.2f, Column : '%s', Threshold : %.2f" % (self.name, data_impurity, impurity, self.col, self.thresh))
        
        if(impurity >= data_impurity and not force_decision):
            self.make_leaf(x.mode()['target'][0])
            self.col    = None
            self.thresh = None

    # Split the data into two, if its not a leaf node
    def split(self, x, drop=False):
        if(self.leaf):
            if(self.debug):
                print(" %15s : Cant split, is a leaf node."%self.name)
            return False

        if(self.debug):
            print(" %15s : Splitting input..."%self.name)
        
        return split_data(x, self.col, self.thresh, drop)
    
    # Attach the upper child node
    def attach_upper(self, upper_node):
        if(self.leaf):
            print(" %15s : Cant attach, is a leaf node"%self.name)
            return False
        
        self.upper = upper_node
        return True

        if(self.debug):
            print(" %15s : Attached upper node '%s'"%(self.name, self.upper.name))

    # Attach the lower child node
    def attach_lower(self, lower_node):
        if(self.leaf):
            print(" %15s : Cant attach, is a leaf node"%self.name)
            return False

        self.lower = lower_node
        return True

        if(self.debug):
            print(" %15s : Attached upper node '%s'"%(self.name, self.lower.name))

    # Classify a given data sample
    # If the node is a leaf node, it returns the assigned state
    # Else, it uses the decision criteria to call either the upper or lower child node.
    def classify(self, x):
        if(self.leaf):
            if(self.debug):
                print(" %15s : Leaf node - returning result %d"%(self.name, self.state))
            return self.state

        if(self.debug):
            print(" %15s : Classifying criteria - %s >= %.2f"%(self.name, self.col, self.thresh))
        if(x[self.col] > self.thresh):
            if(self.debug):
                print(" %15s : Moving to upper node"%self.name)
            return self.upper.classify(x)
        else:
            if(self.debug):
                print(" %15s : Moving to lower node"%self.name)
            return self.lower.classify(x)

    # Create and link upper and lower child nodes
    def make_children(self, debug=None):
        if(self.debug):
            print(" %15s : Making children nodes..."%self.name)

        if(debug==None):
            debug=self.debug

        upper = DecisionNode(self, name='%s+'%(self.name), debug=debug)
        lower = DecisionNode(self, name='%s-'%(self.name), debug=debug)

        self.attach_upper(upper)
        self.attach_lower(lower)

        return upper, lower

    # Returns the upper and lower child nodes
    def get_children(self):
        return self.upper, self.lower

    # Change debug state.
    # Setting propagate to True will propagate the change in debug state down its child nodes.
    def set_debug(self, debug=True, propagate=False):
        if(self.debug != debug):
            print(" %15s : Setting debug to %s"%(self.name, debug))
        self.debug=debug
        if(propagate):
            if(not self.leaf):
                self.upper.set_debug(debug, True)
                self.lower.set_debug(debug, True)

# Recurseive depth-first search, while building the tree
def build_tree(data, level, node=None, debug=False, drop=False, max_tries=None, min_data=None, root_name='root', ):
    # If at the last level, make the node a leaf node
    if(level == 0):
        leaf_out = data.mode()['target'][0]
        node.make_leaf(leaf_out)
        return
    
    # If no parent node is passed, create a new node, called the 'root' node.
    if(node == None):
        node = DecisionNode(None, name=root_name, debug=debug)
    
    # Train the node
    if(min_data == None or data.shape[0] >= min_data):
        node.train(data, max_tries=max_tries)
    else:
        leaf_out = data.mode()['target'][0]
        node.make_leaf(leaf_out)
    
    # If the node decided to be a leaf node, stop building further
    if(node.leaf):
        return

    # Else, make child nodes, and split the dataset between these nodes
    node.make_children()
    upper_ds, lower_ds = node.split(data, drop)
    
    # Build the tree down from these nodes, using the split dataset.
    # Note : Its a depth-first search because the upper node is called first, and only after
    #   its tree has been built, is the lower node called
    build_tree(upper_ds, level-1, node.upper, debug=debug, drop=drop, max_tries=max_tries, min_data=min_data)
    build_tree(lower_ds, level-1, node.lower, debug=debug, drop=drop, max_tries=max_tries, min_data=min_data)

    return node