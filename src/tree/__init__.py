# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:44:51 2022

@author: hzhang3410
"""

import collections


class Tree(object):
    """
        A Python implementation of Tree data structure
    """

    def __init__(self, name=None, children=None, metadata=None):
        """
        @param name: name of this node
        @param children: sub node(s) of Tree, could be None, child (single) or children (multiple)
        @param metadata: metadata associated with this node
        """
        self.name = name
        self.metadata = metadata
        self.__children = []
        self.__parent = None  # private parent attribute

        if children:  # construct a Tree with child or children
            if isinstance(children, Tree):
                self.__children.append(children)
                children.__parent = self

            elif isinstance(children, collections.Iterable):
                for child in children:
                    if isinstance(child, Tree):
                        self.__children.append(child)
                        child.__parent = self
                    else:
                        raise TypeError('_child of Tree should be a Tree type.')
            else:
                raise TypeError('_child of Tree should be a Tree type')

    def __setattr__(self, name, value):

        """
            Hide the __parent and __children attribute from using dot assignment.
            To add __children, please use add_child or add_children method; And
            node's parent isn't assignable
        """

        if name in ('parent', '__parent', 'children'):
            raise AttributeError("To add children, please use add_child or add_children method.")
        else:
            super().__setattr__(name, value)

    def __str__(self, *args, **kwargs):

        return self.name.__str__(*args, **kwargs)

    def add_child(self, child):
        """
            Add one single child node to current node
        """
        if isinstance(child, Tree):
            self.__children.append(child)
            child.__parent = self
        else:
            raise TypeError('_child of Tree should be a Tree type')

    def add_children(self, children):
        """
            Add multiple child nodes to current node
        """
        if isinstance(children, list):
            for child in children:
                if isinstance(child, Tree):
                    self.__children.append(child)
                    child.__parent = self
                else:
                    raise TypeError('_child of Tree should be a Tree type.')

    def get_parent(self):
        """
            Get node's parent node.
        """
        return self.__parent

    def get_child(self, index):
        """
            Get node's No. index child node.
            @param index: Which child node to get in children list, starts with 0 to number of children - 1
            @return:  A Tree node presenting the number index child
            @raise IndexError: if the index is out of range
        """
        try:
            return self.__children[index]
        except IndexError:
            raise IndexError("Index starts with 0 to number of children - 1")

    def get_children(self):
        """
            Get node's all child nodes.
        """
        return self.__children

    def get_node(self, content, include_self=True):
        """

            Get the first matching item(including self) whose data is equal to content.
            Method uses data == content to determine whether a node's data equals to content, note if your node's data
            is self defined class, overriding object's __eq__ might be required.
            Implement Tree travel (level first) algorithm using queue
            If include only child, search only in the child then
            @param content: node's content to be searched
            @param include_self: True
            @return: Return node which contains the same data as parameter content, return None if no such node
        """

        nodes_q = []

        if include_self:
            nodes_q.append(self)
        else:
            nodes_q.extend(self.get_children())

        while nodes_q:
            child = nodes_q[0]
            if child.name == content:
                return child
            else:
                nodes_q.extend(child.get_children())
                del nodes_q[0]

    def get_node_from_children(self, content):
        for child in self.get_children():
            if child.name == content:
                return child

    def del_child(self, index):
        """
            Delete node's No. index child node.
            @param index: Which child node to delete in children list, starts with 0 to number of children - 1
            @raise IndexError: if the index is out of range
        """
        try:
            del self.__children[index]
        except IndexError:
            raise IndexError("Index starts with 0 to number of children - 1")

    def del_node(self, content):

        """
            Delete the first matching item(including self) whose data is equal to content.
            Method uses data == content to determine whether a node's data equals to content, note if your node's data
            is self defined class, overriding object's __eq__ might be required.
            Implement Tree travel (level first) algorithm using queue
            @param content: node's content to be searched
        """

        nodes_q = [self]

        while nodes_q:
            child = nodes_q[0]
            if child.name == content:
                if child.is_root():
                    del self
                    return
                else:
                    parent = child.get_parent()
                    parent.del_child(parent.get_children().index(child))
                    return
            else:
                nodes_q.extend(child.get_children())
                del nodes_q[0]

    def get_root(self):
        """
            Get root of the current node.
        """
        if self.is_root():
            return self
        else:
            return self.get_parent().get_root()

    def is_root(self):
        """
            Determine whether node is a root node or not.
        """
        if self.__parent is None:
            return True
        else:
            return False

    def is_branch(self):
        """
            Determine whether node is a branch node or not.
        """
        if not self.__children:
            return True
        else:
            return False

    def pretty_tree(self):
        """"
            Another implementation of printing tree using Stack
            Print tree structure in hierarchy style.
            For example:
                _root
                |___ C01
                |     |___ C11
                |          |___ C111
                |          |___ C112
                |___ C02
                |___ C03
                |     |___ C31
            A more elegant way to achieve this function using Stack structure,
            for constructing the _nodes Stack push and pop nodes with additional level info.
        """

        level = 0
        _nodes_s = [self, level]  # init _nodes Stack

        while _nodes_s:
            head = _nodes_s.pop()
            # head pointer points to the first item of stack, can be a level identifier or tree node
            if isinstance(head, int):
                level = head
            else:
                self.__print_label__(head, _nodes_s, level)
                children = head.get_children()
                children.reverse()

                if _nodes_s:
                    _nodes_s.append(level)  # push level info if stack is not empty

                if children:  # add children if has children nodes
                    _nodes_s.extend(children)
                    level += 1
                    _nodes_s.append(level)

    def nested_tree(self):
        """"
            Print tree structure in nested-list style.
            For example:
            [0] nested-list style
                [_root[C01[C11[C111,C112]],C02,C03[C31]]]
            """

        nested_t = ''
        delimiter_o = '['
        delimiter_c = ']'
        _nodes_s = [delimiter_c, self, delimiter_o]

        while _nodes_s:
            head = _nodes_s.pop()
            if isinstance(head, str):
                nested_t += head
            else:
                nested_t += str(head.name)

                children = head.get_children()

                if children:  # add children if has children nodes
                    _nodes_s.append(delimiter_c)
                    for child in children:
                        _nodes_s.append(child)
                        _nodes_s.append(',')
                    _nodes_s.pop()
                    _nodes_s.append(delimiter_o)

        print(nested_t)

    @staticmethod
    def __print_label__(head, _nodes_s, level):
        """
           Print each node
        """
        leading = ''
        lasting = '|___ '
        label = str(head.name)

        if level == 0:
            print(str(head))
        else:
            for leaf in range(0, level - 1):
                sibling = False
                parent_t = head.__get_parent__(level - leaf)
                for c in parent_t.get_children():
                    if c in _nodes_s:
                        sibling = True
                        break
                if sibling:
                    leading += '|     '
                else:
                    leading += '     '

            if label.strip() != '':
                print('{0}{1}{2}'.format(leading, lasting, label))

    def __get_parent__(self, up):
        parent = self
        while up:
            parent = parent.get_parent()
            up -= 1
        return parent

