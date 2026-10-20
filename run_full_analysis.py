"""
Master Runner for the Infant EEG Universality and Individuality Pipeline.
Executes all analytical phases sequentially and generates all tables and figures.
"""
import time
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Master Infant EEG Longitudinal Analysis Pipeline")
    parser.add_argument("--start-phase", type=int, default=2, help="Phase number to start from (default: 2)")
    parser.add_argument("--permutations", type=int, default=10000, help="Number of permutations (default: 10,000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    args = parser.parse_args()
    
    t_start = time.time()
    print("=" * 80)
    print("STARTING FULL INFANT EEG LONGITUDINAL PIPELINE (N = 19)")
    print(f"Start Phase: {args.start_phase} | Permutations: {args.permutations} | Workers: {args.workers}")
    print("=" * 80)
    
    if args.start_phase <= 1:
        mod1 = importlib.import_module("analysis_pipeline.01_extract_tensors")
        mod1.run_extraction()
        
    if args.start_phase <= 2:
        mod2 = importlib.import_module("analysis_pipeline.02_compute_universality")
        mod2.run_phase_2(n_permutations=args.permutations, n_workers=args.workers)
        
    if args.start_phase <= 3:
        mod3 = importlib.import_module("analysis_pipeline.03_compute_within_session")
        mod3.run_within_session()
        
    if args.start_phase <= 4:
        mod4 = importlib.import_module("analysis_pipeline.04_compute_identifiability")
        mod4.run_identifiability(n_permutations=args.permutations)
        
    if args.start_phase <= 5:
        mod5 = importlib.import_module("analysis_pipeline.05_taxonomy_and_variance")
        mod5.run_taxonomy_and_variance()
        
    if args.start_phase <= 6:
        mod6 = importlib.import_module("analysis_pipeline.06_developmental_and_twins")
        mod6.run_developmental_and_twins(n_permutations=args.permutations)
        
    if args.start_phase <= 7:
        mod7 = importlib.import_module("analysis_pipeline.07_robustness_checks")
        mod7.run_robustness_checks()
        
    if args.start_phase <= 8:
        mod8 = importlib.import_module("analysis_pipeline.08_export_tables")
        mod8.export_all_tables()
        
    if args.start_phase <= 9:
        mod9 = importlib.import_module("analysis_pipeline.09_generate_figures")
        mod9.generate_all_figures()
        
    t_end = time.time()
    elapsed = t_end - t_start
    print("=" * 80)
    print(f"PIPELINE FULLY COMPLETE IN {elapsed/60:.2f} MINUTES ({elapsed:.1f} SECONDS)!")
    print("All tables saved in: tables/")
    print("All figures saved in: figures/")
    print("All processed tensors saved in: processed_tensors/")
    print("=" * 80)

if __name__ == "__main__":
    main()
