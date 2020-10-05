from simulate.data.create_data import generate_bitstreams, generate_signals
from simulate.simulate_algorithm.simulate_algorithm import simulate_algorithm
from brrn.model.brrn import load_model

def run():
    bitstreams = generate_bitstreams()
    signals = generate_signals(bitstreams)
    model = load_model('freedolite')
    simulate_algorithm(model, signals)