"""
Top level API for database setup and address matching

This module will be made available to module level
"""

from .database_setup import create_database
from .match import match
from .score import score
from .score_and_match import score_and_match