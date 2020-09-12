from .calculus import *
from .cmd import *
from .colors import *
from .constants import *
from .conversion import *
from .counter import *
from .introspection import *
from .io import *
from .logger import *
from .plotting import *
from .progress import *
from .samples import *
from .series import *

#  Instantiate the default argument parser at runtime
command_line_args, command_line_parser = set_up_command_line_arguments()
#  Instantiate the default logging
setup_logger(print_version=False, log_level=command_line_args.log_level)
