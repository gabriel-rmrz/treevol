#!/usr/bin/env python
import os
import sys
import argparse
from config import Configuration

def add_common_config(parser):
  parser.add_argument('config', help='Configuration file')

def add_common_options(parser):
  parser.add_argument('--baseDir',default='', help='Working directory')

def add_common_clustering(parser):
  pass

def add_common_isolation(parser):
  pass

def add_qsm(parser):
  pass

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Trees volume calculation")
  subparsers = parser.add_subparsers(help='Tree volume calculation', dest='command')
  subparsers.required = True

  parser_prepartion = subparsers.add_parser('prepare', help='Prepare the h5 slices files from the .las(laz) input file')
  add_common_config(parser_prepartion)
  add_common_options(parser_prepartion)
  return parser.parse_args(argv)

def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)

  if args.command == 'prepare':
    from prepare_data_for_clustering import prepare
    prepare(args.baseDir,Configuration(args.config))

if __name__ == '__main__':
  status = main()
  sys.exit(status)
