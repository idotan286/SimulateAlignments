from re import split, compile
import os
from Bio import SeqIO

class tree_validator():
	def validate_tree(self,path,tree_filename):
		with open(os.path.join(path, tree_filename), 'r') as f:
			tree = f.read()

		if self.__is_newick(tree):
			self.__fix_tree(path,tree_filename)
			return True
		else: 
			raise ValueError("Error: invalid tree file. Newick format is required.")

	# returns true if the given input is a number
	def __is_number(self,string):
		try:
			float(string)
			return True
		except ValueError:
			return False


	# returns the index of the first comma outside parenthesis
	# comma separates branches
	def __find_branch(self,parsed_tokens):
		open = 0
		closed = 0
		first_comma_index = None

		for char_index, char in enumerate(parsed_tokens):
			if char == "(":
				open += 1
			elif char == ")":
				closed += 1

			if open - closed == 0:
				if char == "," \
						and first_comma_index is None:
					first_comma_index = char_index

		return first_comma_index


	# tree -> subtree [label] [: length] ";"
	# tree -> subtree ";" | branch ";"
	def __parse_branch(self,parsed_tokens):
		if self.__is_number(parsed_tokens[-1]):
			# found a branch with length
			return self.__parse_branch(parsed_tokens)

		# subtree without label and length
		return self.__parse_subtree(parsed_tokens)


	# subtree -> leaf | internal
	def __parse_subtree(self,parsed_tokens):
		try:
			if parsed_tokens[0] == '(':

				# found an internal node
				if parsed_tokens[-1] == ')':
					return self.__parse_internal(parsed_tokens)

				if ')' not in parsed_tokens:
					print("Unbalanced parentheses in %s!" % ''.join(parsed_tokens))
					return False

				else:

					if self.__parse_name(parsed_tokens[-1]):
						# found a labelled internal node
						return self.__parse_internal(parsed_tokens[:-1])

					else:
						return False

			else:

				if ')' in parsed_tokens:
					print("Unbalanced parentheses in %s!" % ''.join(parsed_tokens))
					return False

				# found a leaf
				return self.__parse_name(parsed_tokens[0])

		except IndexError:
			pass


	# leaf --> name
	# name --> empty | string
	def __parse_name(self,name):

		# checking whether a string contains a space
		if ' ' in name:
			print("Error: space in %s." % name)
			return False

		# checking whether a string contains :
		if ':' in name:
			print("Error: colon in %s." % name)
			return False

		# checking whether a string contains (
		if '(' in name or ')' in name:
			print("Error: unbalanced parentheses in %s." % name)
			return False

		# checking whether a string contains ;
		if ';' in name:
			print("Error: semicolon in %s." % name)
			return False

		return True


	# branchset --> branch | branch "," branchset
	def __parse_branchset(self,parsed_tokens):
		comma = self.__find_branch(parsed_tokens)

		if comma is None:
			# found a single branch
			return self.__parse_branch(parsed_tokens)

		# found a branch and a branchset
		else:

			if self.__parse_branch(parsed_tokens[0:comma]):
				# successful parsing
				return self.__parse_branchset(parsed_tokens[comma + 1:])

			else:
				return False


	# branch --> subtree length
	def __parse_branch(self,parsed_tokens):
		# empty branch
		if not parsed_tokens:
			return True

		# length is not empty
		try:
			if parsed_tokens[-2] == ':':
				length_ok = self.__parse_length(parsed_tokens[-1])

				# label or subtree are not empty
				if parsed_tokens[:-2]:
					subtree_ok = self.__parse_subtree(parsed_tokens[:-2])
					return length_ok and subtree_ok

				else:
					return length_ok

		except IndexError:
			pass

		# there is only a subtree
		return self.__parse_subtree(parsed_tokens)


	# length --> empty | ":" number
	def __parse_length(self,number):
		if self.__is_number(number):
			return True

		print("%s is not a number." % number)
		return False


	# internal --> "(" branchset ")" name
	def __parse_internal(self,parsed_tokens):
		if parsed_tokens[-1] != ')':
			# name is not empty
			name_ok = self.__parse_name(parsed_tokens[-1])

			if name_ok:
				return self.__parse_branchset(parsed_tokens[1:-2])

			else:
				return False

		# controls on balanced parentheses already made
		return self.__parse_branchset(parsed_tokens[1:-1])


	# first function performing the initial controls
	def __is_newick(self,tree):
		# dividing the string into tokens, to check them singularly
		tokens = split(r'([A-Za-z]+[^A-Za-z,)]+[A-Za-z]+|[0-9.]*[A-Za-z]+[0-9.]+|[0-9.]+\s+[0-9.]+|[0-9.]+|[A-za-z]+|\(|\)|;|:|,)', tree)

		# removing spaces and empty strings (spaces within labels are still present)
		parsed_tokens = list(filter(lambda x: not (x.isspace() or not x), tokens))

		# checking whether the tree ends with ;
		if parsed_tokens[-1] != ';':
			print("Tree without ; at the end.")
			return False

		# first controls passed, calling the recursive function
		else:
			del parsed_tokens[-1]
			return self.__parse_branch(parsed_tokens)



	def __scientific_to_decimal(self, expression):
		return f"{float(expression):.10f}"

	def __remove_bootstrap(self, expression):
		if expression[0] == ")":
			return "):"
		else:
			return ""

	def __fix_tree(self, file_path, file_name):
		scinot_regex = compile(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)')
		bootstrap_regex = compile(r'\)(\d+(?:\.\d+)?):|:(\d+(?:\.\d+)?)\[(\d+(?:\.\d+)?)\]')
		with open(file_path+file_name, 'r') as f:
			tree = f.read()

		new_tree = scinot_regex.sub(lambda x: self.__scientific_to_decimal(x.group()), tree)
		new_tree = bootstrap_regex.sub(lambda x: self.__remove_bootstrap(x.group()), new_tree)

		with open(file_path+file_name, 'w') as f:
			f.write(new_tree)


class msa_validator():

	def validate_msa(self,path,msa_filename, mode):
		msa = list(SeqIO.parse(os.path.join(path,msa_filename),'fasta'))
		if self.__is_mode_fit(msa, mode):
			return True
		else:
			raise ValueError("Error: invalid MSA file. Fasta format is required.")
	
	def __is_mode_fit(self, msa, mode):
		if mode == "amino":
			return True
		if mode == "nuc":
			amino_regex = compile('[^-ATCGatcg]')
			for i in msa:

				if amino_regex.search(str(i.seq)):
					print(f"Invalid character in organism '{i.id}'. When running in 'nuc' mode make sure the MSA contains only nucleotides")
					return False
		return True