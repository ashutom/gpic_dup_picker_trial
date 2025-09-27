"""
DupFinder Logging Module

This module provides comprehensive logging functionality with three types of logs:
1. Console log - controlled by console_level (what user sees)
2. Application log - controlled by app_level (structured debugging)  
3. Journal log - always DEBUG level (captures everything for diagnosis)
"""

import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir=None, console_level=logging.INFO, app_level=logging.INFO):
    """
    Setup comprehensive logging configuration with three types of logs:
    1. Console log - controlled by console_level
    2. Application log - controlled by app_level  
    3. Journal log - always DEBUG level (captures everything)
    
    Args:
        log_dir (str): Directory to store log files
        console_level (int): Logging level for console output
        app_level (int): Logging level for application log file
    """
    # Create logs directory if specified
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        app_log_file = os.path.join(log_dir, 'dupfinder_app.log')
        journal_log_file = os.path.join(log_dir, 'dupfinder_journal.log')
    else:
        app_log_file = 'dupfinder_app.log'
        journal_log_file = 'dupfinder_journal.log'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to lowest level to capture everything
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Define log formats
    detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    simple_format = '%(levelname)s - %(funcName)s - %(message)s'
    journal_format = '%(asctime)s - %(process)d - %(thread)d - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
    
    # 1. Console Handler - controlled by console_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(simple_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 2. Application Log Handler - controlled by app_level
    app_handler = RotatingFileHandler(
        app_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    app_handler.setLevel(app_level)
    app_formatter = logging.Formatter(detailed_format)
    app_handler.setFormatter(app_formatter)
    logger.addHandler(app_handler)
    
    # 3. Journal Log Handler - always DEBUG level (captures everything)
    journal_handler = RotatingFileHandler(
        journal_log_file, 
        maxBytes=50*1024*1024,  # 50MB for journal
        backupCount=10
    )
    journal_handler.setLevel(logging.DEBUG)  # Always capture everything
    journal_formatter = logging.Formatter(journal_format)
    journal_handler.setFormatter(journal_formatter)
    logger.addHandler(journal_handler)
    
    logging.debug("Advanced logging system initialized")
    logging.debug(f"Console level: {logging.getLevelName(console_level)}")
    logging.debug(f"Application log level: {logging.getLevelName(app_level)}")
    logging.debug(f"Journal log level: DEBUG (captures everything)")
    logging.debug("This is a debug message that should appear in journal log")


def get_log_levels():
    """
    Return dictionary of available log levels for easy reference
    
    Returns:
        dict: Dictionary mapping level names to logging constants
    """
    return {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }


def set_console_level(level):
    """
    Dynamically change console logging level
    
    Args:
        level (int): New logging level for console
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler.setLevel(level)
            logging.info(f"Console log level changed to: {logging.getLevelName(level)}")
            break


def set_app_level(level):
    """
    Dynamically change application log level
    
    Args:
        level (int): New logging level for application log
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and 'app.log' in handler.baseFilename:
            handler.setLevel(level)
            logging.info(f"Application log level changed to: {logging.getLevelName(level)}")
            break


def log_system_info():
    """Log system information for debugging purposes"""
    import sys
    import platform
    
    logging.debug("=== System Information ===")
    logging.debug(f"Python version: {sys.version}")
    logging.debug(f"Platform: {platform.platform()}")
    logging.debug(f"Architecture: {platform.architecture()}")
    logging.debug(f"Machine: {platform.machine()}")
    logging.debug(f"Processor: {platform.processor()}")
    logging.debug("=== End System Information ===")


def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.debug(f"Memory usage - RSS: {memory_info.rss / 1024 / 1024:.2f} MB, "
                     f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        logging.debug("psutil not available for memory monitoring")
    except Exception as e:
        logging.debug(f"Error getting memory usage: {e}")


def cleanup_logs(log_dir=None, days_to_keep=30):
    """
    Clean up old log files
    
    Args:
        log_dir (str): Directory containing log files
        days_to_keep (int): Number of days of logs to keep
    """
    import time
    
    if not log_dir:
        log_dir = os.getcwd()
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
    
    try:
        for filename in os.listdir(log_dir):
            if filename.endswith('.log') or '.log.' in filename:
                file_path = os.path.join(log_dir, filename)
                file_time = os.path.getctime(file_path)
                
                if file_time < cutoff_time:
                    os.remove(file_path)
                    logging.info(f"Cleaned up old log file: {filename}")
    
    except Exception as e:
        logging.error(f"Error during log cleanup: {e}")


def test_logging_system(log_dir=None):
    """
    Test function to verify all logging levels and handlers work correctly
    
    Args:
        log_dir (str): Directory to store test log files
    """
    print("Testing logging system...")
    
    # Setup logging with different levels for testing
    setup_logging(
        log_dir=log_dir,
        console_level=logging.INFO,  # Console shows INFO and above
        app_level=logging.DEBUG      # App log shows DEBUG and above
    )
    
    # Test all logging levels
    logging.debug("This is a DEBUG message - should appear in journal and app logs only")
    logging.info("This is an INFO message - should appear in all logs")
    logging.warning("This is a WARNING message - should appear in all logs")
    logging.error("This is an ERROR message - should appear in all logs")
    logging.critical("This is a CRITICAL message - should appear in all logs")
    
    # Test system info logging
    log_system_info()
    
    # Test memory usage logging
    log_memory_usage()
    
    # Test dynamic level changes
    print("\nTesting dynamic level changes...")
    set_console_level(logging.WARNING)  # Now console should only show WARNING and above
    
    logging.info("This INFO message should NOT appear on console after level change")
    logging.warning("This WARNING message SHOULD appear on console after level change")
    
    # Reset console level back
    set_console_level(logging.INFO)
    logging.info("Console level reset - this INFO message should appear again")
    
    # Verify log files were created
    if log_dir:
        app_log_path = os.path.join(log_dir, 'dupfinder_app.log')
        journal_log_path = os.path.join(log_dir, 'dupfinder_journal.log')
    else:
        app_log_path = 'dupfinder_app.log'
        journal_log_path = 'dupfinder_journal.log'
    
    print(f"\nChecking log files:")
    print(f"Application log exists: {os.path.exists(app_log_path)}")
    print(f"Journal log exists: {os.path.exists(journal_log_path)}")
    
    if os.path.exists(app_log_path):
        with open(app_log_path, 'r', encoding='utf-8') as f:
            app_lines = f.readlines()
            print(f"Application log has {len(app_lines)} lines")
    
    if os.path.exists(journal_log_path):
        with open(journal_log_path, 'r', encoding='utf-8') as f:
            journal_lines = f.readlines()
            print(f"Journal log has {len(journal_lines)} lines")
    
    print("\nLogging test completed!")
    return True


def verify_logging_configuration():
    """
    Verify that the logging configuration is set up correctly
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    logger = logging.getLogger()
    
    # Check if logger level is set to DEBUG
    if logger.level != logging.DEBUG:
        print(f"ERROR: Logger level is {logging.getLevelName(logger.level)}, should be DEBUG")
        return False
    
    # Check number of handlers
    if len(logger.handlers) != 3:
        print(f"ERROR: Expected 3 handlers, found {len(logger.handlers)}")
        return False
    
    # Verify handler types
    handler_types = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler_types.append("console")
        elif isinstance(handler, RotatingFileHandler):
            if 'app.log' in handler.baseFilename:
                handler_types.append("application")
            elif 'journal.log' in handler.baseFilename:
                handler_types.append("journal")
    
    expected_handlers = ["console", "application", "journal"]
    if sorted(handler_types) != sorted(expected_handlers):
        print(f"ERROR: Expected handlers {expected_handlers}, found {handler_types}")
        return False
    
    print("Logging configuration verified successfully!")
    return True


if __name__ == "__main__":
    """Run logging tests when module is executed directly"""
    print("Running dupflogging module tests...")
    
    # Test in current directory
    test_logging_system()
    
    # Verify configuration
    verify_logging_configuration()
    
    print("\nAll tests completed!")
