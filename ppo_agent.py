"""PPO trains inside the physical closed loop with strict checkpoint signatures."""

from navstack.experiments import train as train_agent, load_run as load_agent, evaluate

if __name__ == "__main__":
    from navstack.cli import main

    main()
