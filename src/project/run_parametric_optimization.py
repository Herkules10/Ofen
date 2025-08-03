"""
Runner script for parametric optimization of neural architecture search algorithms.
This script executes the grid search with the specified parameter spaces.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project directory to Python path
current_dir = Path(__file__).parent
project_dir = current_dir
sys.path.insert(0, str(project_dir))

def run_parametric_optimization():
    """Run the parametric optimization with error handling."""
    try:
        from parametric_optimization import ParametricOptimizer
        
        print("🚀 Starting Parametric Optimization")
        print("="*50)
        
        # Create optimizer with project directory
        optimizer = ParametricOptimizer()
        
        # Run optimization
        optimizer.optimize_all_algorithms()
        
        print("\n✅ Parametric optimization completed successfully!")
        print("📁 Results saved to: parametric_results/")
        
        # Run analysis if requested
        try:
            from parametric_analysis import EnhancedParametricAnalyzer
            
            print("\n📊 Running analysis...")
            analyzer = EnhancedParametricAnalyzer("parametric_results", "parametric_figures")
            analyzer.analyze_all()
            
            print("📁 Analysis plots saved to: parametric_figures/")
            
        except Exception as e:
            print(f"⚠️ Analysis failed: {e}")
            print("You can run analysis separately using parametric_analysis.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available in the project directory")
        return False
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = [
        'torch', 'torchvision', 'numpy', 'matplotlib', 
        'seaborn', 'pandas', 'pickle', 'json'
    ]
    
    optional_modules = [
        'pynvml'  # For GPU monitoring (optional)
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    # Check optional modules but don't fail if missing
    missing_optional = []
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing:
        print(f"❌ Missing required modules: {', '.join(missing)}")
        print("Please install them using: pip install torch torchvision numpy matplotlib seaborn pandas")
        return False
    
    if missing_optional:
        print(f"⚠️ Missing optional modules: {', '.join(missing_optional)}")
        print("GPU monitoring will be disabled. Install with: pip install pynvml")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run Parametric Optimization')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run analysis only (skip optimization)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    if args.check_deps:
        print("✅ All dependencies are available!")
        return
    
    # Run analysis only if requested
    if args.analysis_only:
        try:
            from parametric_analysis import EnhancedParametricAnalyzer
            
            print("📊 Running analysis only...")
            analyzer = EnhancedParametricAnalyzer("parametric_results", "parametric_figures")
            analyzer.analyze_all()
            
            print("📁 Analysis plots saved to: parametric_figures/")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        return
    
    # Run full optimization
    success = run_parametric_optimization()
    
    if success:
        print("\n🎉 Parametric optimization pipeline completed!")
        print("\nNext steps:")
        print("1. Check parametric_results/ for detailed results")
        print("2. Check parametric_figures/ for analysis plots")
        print("3. Review the best parameter configurations")
        print("4. Use the best parameters for final experiments")
    else:
        print("\n💥 Parametric optimization failed!")
        print("Check the error messages above for details")

if __name__ == "__main__":
    main()
