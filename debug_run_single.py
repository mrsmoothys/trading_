#!/usr/bin/env python
"""
Debug wrapper for run_single.py that catches and reports any exceptions.
"""
import os
import sys
import traceback
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('debug_run_single.log')]
)
logger = logging.getLogger("DebugWrapper")

def main():
    """Run the run_single.py script with exception handling."""
    logger.info("Starting debug wrapper for run_single.py")
    
    # Print environment info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Command line arguments: {sys.argv}")
    
    # Check if run_single.py exists
    if not os.path.exists('run_single.py'):
        logger.error("run_single.py not found in current directory")
        return 1
    
    # Create modified sys.argv for run_single.py
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0].replace('debug_run_single.py', 'run_single.py')] + sys.argv[1:]
    
    try:
        # Import run_single as a module
        logger.info("Attempting to import run_single")
        import run_single
        
        # If run_single has a main function, call it directly
        if hasattr(run_single, 'main'):
            logger.info("Calling run_single.main()")
            run_single.main()
        else:
            logger.warning("No main() function found in run_single.py")
            logger.info("Executing run_single.py content")
            # Execute the file directly
            with open('run_single.py') as f:
                code = compile(f.read(), 'run_single.py', 'exec')
                exec(code, {})
                
        logger.info("run_single.py completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())