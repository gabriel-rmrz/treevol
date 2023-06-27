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

def make_dir_structure(baseDir):
  if not os.path.exists(baseDir):
    print("Creating directory '{}'...".format(baseDir))
    os.makedirs(baseDir)
    print("Creating directory '{}/data'...".format(baseDir))
    os.makedirs(baseDir+'/data')
    print("Creating directory '{}/plots'...".format(baseDir))
    os.makedirs(baseDir+'/plots')
    print("Creating directory '{}/results'...".format(baseDir))
    os.makedirs(baseDir+'/results')
    print("Creating directory '{}/scripts'...".format(baseDir))
    os.makedirs(baseDir+'/scripts')
  else:
    print("Directory '{}' already exists. Please delete it or change it the value of baseDir".format(baseDir))
    return 0
  return 1

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Trees volume calculation")
  subparsers = parser.add_subparsers(help='Tree volume calculation', dest='command')
  subparsers.required = True

  parser_prepartion = subparsers.add_parser('prepare', help='Prepare the h5 slices files from the .las(laz) input file')
  add_common_config(parser_prepartion)
  add_common_options(parser_prepartion)

  parser_clustering = subparsers.add_parser('clustering', help='Use clustering algorithm si isolate individual trees')
  add_common_config(parser_clustering)
  add_common_options(parser_clustering)
  parser_eff = subparsers.add_parser('calc_eff', help='Compute efficiency for the different algorithms')
  add_common_config(parser_eff)
  add_common_options(parser_eff)

  return parser.parse_args(argv)

def main(argv=None):
  if argv is None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)

  if args.command == 'prepare':
    from prepare_data_for_clustering import prepare
    if not make_dir_structure(args.baseDir):
      return 0
    prepare(args.baseDir,Configuration(args.config))
  elif args.command == 'clustering':
    from clustering import clustering
    clustering(args.baseDir,Configuration(args.config))
  elif args.command == 'calc_eff':
    from calc_eff import calc_eff
    calc_eff(args.baseDir,Configuration(args.config))

if __name__ == '__main__':
  status = main()
  sys.exit(status)
