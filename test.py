import argparse
parser = argparse.ArgumentParser(description='You can add a description here')
parser.add_argument('-n','--name', help='Your name',required=False)
parser.add_argument('-q', '--query')
args = parser.parse_args()


if __name__ == '__main__':
	print (args.name, args.query)
