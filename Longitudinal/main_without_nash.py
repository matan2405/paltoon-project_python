"""
Main entry point for the refactored platoon simulation system.
This file demonstrates the new modular structure and provides the same functionality
as the original monolithic file.
"""

import os
import sys
import time
import numpy as np

# Import configuration and modules
from config import setup_matplotlib, HEADLESS_MODE
from simulation.simulator import PlatoonSimulation, run_simulation
from visualization.animation import create_platoon_animation
from visualization.plots import create_comprehensive_plots, create_detailed_scenario_summary
# Ensure 'vehicle' module is discoverable
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicle"))
from vehicle import Vehicle
from control.human_driver import HumanDriver
from control.platoon_control import PlatoonManager

os.system('cls' if os.name == 'nt' else 'clear')

def run_scenario_join_before():
    """Scenario 1: Join Before platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 1: Join Before platoon")
    print("Vehicle starts ahead, slows down and joins at the front of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = 100.0  # Start ahead
    sim.human_vehicle.state.y = -2.0   # In left lane
    sim.human_driver.target_speed = 100.0 / 3.6  # Lower speed (100 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   🚛 platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_BEFORE", join_trigger_time=25.0)
    return sim


def run_scenario_join_middle():
    """Scenario 2: Join Middle of platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 2: Join Middle of platoon")
    print("Vehicle penetrates into gap in the middle of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = 0.0  # In middle position
    sim.human_vehicle.state.y = -2.0   # In left lane
    sim.human_driver.target_speed = 80.0 / 3.6  # High speed for penetration (80 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   🚛 platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_MIDDLE", join_trigger_time=20.0)
    return sim


def run_scenario_join_after():
    """Scenario 3: Join After platoon"""
    print("\n" + "="*60)
    print("🎯 Scenario 3: Join After platoon")
    print("Vehicle starts from behind, accelerates and joins at the back of platoon")
    print("="*60)
    
    # Create adapted simulation
    sim = PlatoonSimulation()
    
    # Scenario adaptations
    sim.human_vehicle.state.x = -100.0  # Start from behind
    sim.human_vehicle.state.y = -2.0    # In left lane
    sim.human_driver.target_speed = 50.0 / 3.6  # High speed to catch up (50 km/h)

    print(f"🚗 Scenario settings:")
    print(f"   📍 Initial human vehicle position: ({sim.human_vehicle.state.x:.1f}, {sim.human_vehicle.state.y:.1f})")
    print(f"   🎯 Human vehicle target speed: {sim.human_driver.target_speed*3.6:.0f} km/h")
    print(f"   🚛 platoon target speed: {sim.platoon_manager.target_velocity*3.6:.0f} km/h")
    
    # Run simulation
    run_single_simulation(sim, "JOIN_AFTER", join_trigger_time=15.0)
    return sim


def run_single_simulation(sim, scenario_name, join_trigger_time=20.0):
    """Run single simulation"""
    print(f"\n🚀 Starting simulation {scenario_name}")
    
    # Simulation parameters
    max_time = sim.T_sim  # 180 seconds
    T = np.arange(0, max_time, sim.dt)
    max_iterations = len(T)
    human_joined = False
    
    print(f"⏱️ Simulation time: {max_time:.0f} seconds")
    print(f"🔢 Number of iterations: {max_iterations:,}")
    
    start_time = time.time()
    
    try:
        for iteration, t in enumerate(T):
            # Report progress every 2000 iterations
            if iteration % 2000 == 0:
                progress = (iteration / max_iterations) * 100
                print(f"📈 Progress: {progress:5.1f}% (t={sim.time:6.1f}s)")
            
            # Update simulation
            sim.update()
            
            # Trigger joining at the right time
            if (sim.time >= join_trigger_time and 
                not human_joined and not sim.human_driver.merging):
                
                print(f"\n🚨 Activating joining at t={sim.time:.1f}s")
                print(f"📍 Human vehicle position: x={sim.human_vehicle.state.x:.1f}m")
                print(f"📍 platoon positions: {[f'{v.state.x:.1f}m' for v in sim.platoon_vehicles]}")
                
                # Activate joining
                sim.human_driver.merging = True
                human_joined = True
                sim.human_vehicle.joined_platoon = True
                sim.platoon_manager.add_vehicle(sim.human_vehicle)
                sim.human_vehicle.target_velocity = sim.platoon_manager.target_velocity
                sim.human_vehicle.target_acceleration = sim.platoon_manager.max_acceleration
                
                print(f"✅ Human vehicle started joining the platoon")

            # Status report every 5 seconds
            if sim.time % 5.0 < sim.dt:
                # print(f"\n⏰ Time: {sim.time:.1f}s")
                # for vehicle in sim.all_vehicles:
                #     # Determine correct status based on actual vehicle state
                #     if vehicle.vehicle_id.startswith("Platoon"):
                #         status = "platoon"
                #     elif vehicle.vehicle_id == "Human":
                #         if vehicle.joined_platoon:
                #             if sim.human_driver.merging and sim.human_driver.lane_change_progress < 1.0:
                #                 status = "Joining platoon"
                #             else:
                #                 status = "platoon Member"
                #         else:
                #             status = "Independent"
                #     else:
                #         status = "Unknown"
                    
                #     print(f"   {vehicle.vehicle_id}: position=({vehicle.state.x:.1f}, {vehicle.state.y:.1f}), "
                #           f"speed={vehicle.state.vx*3.6:.0f}km/h ({status}), "
                #           f"acceleration={vehicle.state.ax:.2f} m/s²")
                sim.print_status()

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display graphs and animation after scenario completion
        print(f"\n📊 Displaying graphs and animation for {scenario_name}...")
        
        # Create comprehensive plots
        static_plots_fig = create_comprehensive_plots(sim, scenario_name)
        
        # Keep the static plots active but temporary disable interaction during animation
        import time as time_module
        print("⏳ Waiting for static plots to stabilize...")
        time_module.sleep(3)  # Give static plots time to fully render
        
        # Create animation
        try:
            print(f"🎬 Creating animation for {scenario_name}...")
            # Temporarily turn off interactive mode for stability
            import matplotlib.pyplot as plt
            plt.ioff()
            anim = create_platoon_animation(sim, f"{scenario_name} Animation")
            if anim:
                # Keep animation reference to prevent garbage collection warning
                sim._last_animation = anim
                print(f"✅ Animation created successfully")
            else:
                print("⚠️ Animation creation skipped or failed")
                
            # Re-enable interactive mode and refresh static plots
            plt.ion()
            if static_plots_fig:
                plt.figure(static_plots_fig.number)  # Select static plots figure
                plt.draw()
                static_plots_fig.canvas.flush_events()
                print("🔄 Static plots refreshed after animation")
                
        except Exception as anim_error:
            print(f"❌ Animation failed: {anim_error}")
            # Re-enable interactive mode even if animation failed
            import matplotlib.pyplot as plt
            plt.ion()
        
        # Create detailed summary for the scenario
        create_detailed_scenario_summary(sim, scenario_name, execution_time)
        
        # Clean up matplotlib to prevent memory issues
        try:
            import time as time_module
            time_module.sleep(1)
            
            import matplotlib.pyplot as plt
            for i in plt.get_fignums():
                try:
                    fig = plt.figure(i)
                    fig.clf()  # Clear figure content instead of closing
                except:
                    pass
                    
            import gc
            gc.collect()
        except Exception as cleanup_error:
            if "Tcl command" not in str(cleanup_error) and "TclError" not in str(cleanup_error):
                print(f"⚠️ Cleanup note: {cleanup_error}")
        
    except Exception as e:
        print(f"❌ Error in simulation {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")


def run_all_scenarios_separately():
    """Run all scenarios separately"""
    print("🚛 platoon Joining Scenarios System")
    print("=" * 60)
    
    scenarios = [
        ("Scenario 1: Join Before platoon", run_scenario_join_before),
        ("Scenario 2: Join Middle of platoon", run_scenario_join_middle),
        ("Scenario 3: Join After platoon", run_scenario_join_after)
    ]
    
    results = []
    
    for i, (name, scenario_func) in enumerate(scenarios, 1):
        print(f"\n{'🎬' * 20}")
        print(f"Starting {name} [{i}/3]")
        print(f"{'🎬' * 20}")
        
        try:
            result = scenario_func()
            results.append((name, result))
            print(f"✅ {name} completed successfully!")
            
            # Pause between scenarios
            if i < len(scenarios):
                print("\n⏸️ 3-second pause before next scenario...")
                time.sleep(3)
                
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, None))
    
    # Overall summary
    print(f"\n{'🏁' * 25}")
    print("All Scenarios Summary")
    print(f"{'🏁' * 25}")
    
    successful_scenarios = sum(1 for _, result in results if result is not None)
    
    print(f"📊 Overall statistics:")
    print(f"   ✅ Completed scenarios: {successful_scenarios}/3")
    print(f"   📈 Success rate: {successful_scenarios/3*100:.0f}%")
    
    for name, result in results:
        if result:
            human_speed = result.human_vehicle.state.vx * 3.6
            target_speed = result.platoon_manager.target_velocity * 3.6
            joined = "Yes" if result.human_vehicle.joined_platoon else "No"
            print(f"\n🎯 {name}:")
            print(f"   🏎️ Final speed: {human_speed:.1f} km/h (target: {target_speed:.1f})")
            print(f"   📍 Final position: ({result.human_vehicle.state.x:.1f}, {result.human_vehicle.state.y:.1f})")
            print(f"   ✅ Joined platoon: {joined}")
        else:
            print(f"\n❌ {name}: Failed")
    
    return results


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("🚗 Choose what to run:")
    print("1. Single scenario - Join before platoon")
    print("2. Single scenario - Join middle of platoon") 
    print("3. Single scenario - Join after platoon")
    print("4. All three scenarios one after another")
    print("5. Original simulation")
    print("6. Switch to interactive plotting mode (may cause TkAgg errors)")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "6":
            # Switch to interactive mode
            try:
                import matplotlib
                matplotlib.use('Qt5Agg')
                print("✅ Switched to interactive Qt5Agg backend")
                print("⚠️ Note: This may cause destruction errors on exit")
                print("\nNow choose an option to run:")
                print("1-5. Run scenarios with interactive plots")
                choice = input("Enter choice (1-5): ").strip()
            except Exception as backend_error:
                print(f"❌ Could not switch backend: {backend_error}")
                exit(1)
        
        if choice == "1":
            run_scenario_join_before()
        elif choice == "2":
            run_scenario_join_middle()
        elif choice == "3":
            run_scenario_join_after()
        elif choice == "4":
            run_all_scenarios_separately()
        elif choice == "5":
            # Original simulation
            print("🚗 Starting Original Vehicle Platoon Merging Simulation")
            print("=" * 50)
            simulation = run_simulation()
            print("\n📊 Original simulation completed!")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted the program")
    except Exception as e:
        print(f"\n❌ Error: {e}")