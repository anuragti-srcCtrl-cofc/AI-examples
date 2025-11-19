import random
from typing import List, Tuple

def fitness(word: str, target: str) -> int:
    """Calculate fitness as number of matching characters in correct positions."""
    return sum(1 for i, char in enumerate(word) if i < len(target) and char == target[i])

def mutate(word: str, gene_pool: List[str]) -> str:
    """Randomly change one character to a character from one of the gene pool words."""
    word_list = list(word)
    pos = random.randint(0, len(word_list) - 1)
    # Pick a random character from the gene pool
    source_word = random.choice(gene_pool)
    word_list[pos] = random.choice(source_word)
    return ''.join(word_list)

def crossover(word1: str, word2: str) -> str:
    """Combine two words by taking characters from each randomly."""
    length = len(word1)
    child = ''.join(random.choice([word1[i], word2[i]]) for i in range(length))
    return child

def genetic_algorithm(target: str, gene_pool: List[str], population_size: int = 20, generations: int = 1000):
    """
    Evolve a population toward the target word.
    
    Args:
        target: The word we're trying to reach
        gene_pool: List of words to draw characters from
        population_size: Number of individuals in each generation
        generations: Maximum number of generations to run
    
    Returns:
        The best word found and the generation it was found
    """
    # Initialize population with random words from gene pool
    population = [random.choice(gene_pool) for _ in range(population_size)]
    
    for generation in range(generations):
        # Calculate fitness for each individual
        fitness_scores = [(word, fitness(word, target)) for word in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_word, best_score = fitness_scores[0]
        
        print(f"Gen {generation:4d}: {best_word} (fitness: {best_score}/{len(target)})")
        
        # Check if we found the target
        if best_word == target:
            print(f"\nFound target '{target}' at generation {generation}!")
            return best_word, generation
        
        # Selection: keep top performers
        survivors = [word for word, score in fitness_scores[:population_size // 2]]
        
        # Create new population through crossover and mutation
        new_population = survivors.copy()
        while len(new_population) < population_size:
            if random.random() < 0.7:  # 70% chance of crossover
                parent1, parent2 = random.sample(survivors, 2)
                child = crossover(parent1, parent2)
            else:  # 30% chance of mutation only
                child = random.choice(survivors)
            
            child = mutate(child, gene_pool)
            new_population.append(child)
        
        population = new_population
    
    print(f"\nDid not find target within {generations} generations")
    return fitness_scores[0][0], generations

if __name__ == "__main__":
    target = "cat"
    gene_pool = ["bad", "car", "dog", "hat", "pea"]
    
    print(f"Target: {target}")
    print(f"Gene pool: {gene_pool}\n")
    
    best_word, gen = genetic_algorithm(target, gene_pool)
    print(f"\nBest result: {best_word}")