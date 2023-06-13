import os
import sys
import argparse

def add_common_config(parser):
  parser.add_argument('config', help='Configuration file'
def add_common_wd(parser):
  parser.add_argument('wd', help='Working directory'
def add_common_clustering(parser):
  pass

def add_common_isolation(parser):
  pass

def add_qsm(parser):
  pass

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Trees volume calculation")
  subparsers = parser.add_subparsers(help='Tree volume calculation step', dest='command')
  subparsers.required = True

  parser_prepartion = subparsers.add_parser('covert', help='Prepare the h5 slices files from the .las(laz) input file')
def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]
    args = parse_command_line(argv)
  pass
if __name__ == '__main__':
  status = main()
  sys.exit(status)
