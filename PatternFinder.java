import java.util.*;

public class PatternFinder {

    private static final char[][] GRID = {
        {'A', 'B', 'C'},
        {'D', 'E', 'F'},
        {'G', 'H', 'I'}
    };
    
    private static final int SIZE = GRID.length;
    private static final Map<Character, int[]> POSITION_MAP = new HashMap<>();

    static {
        for (int r = 0; r < SIZE; r++) {
            for (int c = 0; c < SIZE; c++) {
                POSITION_MAP.put(GRID[r][c], new int[]{r, c});
            }
        }
    }

    public static void main(String[] args) {
        String start = "A";
        String middle = "I";
        String end = "C";
        
        List<String> patterns = generatePatterns(start, middle, end);
        for (String pattern : patterns) {
            System.out.println(pattern);
        }
    }
    
    public static List<String> generatePatterns(String start, String middle, String end) {
        int populationSize = 100;
        int maxGenerations = 1000;
        double mutationRate = 0.01;
        
        List<String> population = initializePopulation(start, middle, end, populationSize);
        
        for (int gen = 0; gen < maxGenerations; gen++) {
            Collections.sort(population, new Comparator<String>() {
                @Override
                public int compare(String p1, String p2) {
                    return Integer.compare(evaluatePattern(p1, start, middle, end), evaluatePattern(p2, start, middle, end));
                }
            });
            
            List<String> newPopulation = new ArrayList<>();
            
            newPopulation.addAll(population.subList(0, populationSize / 10));
            
            while (newPopulation.size() < populationSize) {
                String parent1 = selectParent(population);
                String parent2 = selectParent(population);
                String child = crossover(parent1, parent2, start, middle, end);
                newPopulation.add(child);
            }
            
            for (int i = 0; i < newPopulation.size(); i++) {
                if (Math.random() < mutationRate) {
                    newPopulation.set(i, mutate(newPopulation.get(i)));
                }
            }
            
            population = newPopulation;
        }
        
        Collections.sort(population, new Comparator<String>() {
            @Override
            public int compare(String p1, String p2) {
                return Integer.compare(evaluatePattern(p1, start, middle, end), evaluatePattern(p2, start, middle, end));
            }
        });
        
        return population;
    }
    
    private static List<String> initializePopulation(String start, String middle, String end, int size) {
        List<String> population = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            String pattern = generateRandomPattern(start, middle, end);
            population.add(pattern);
        }
        return population;
    }
    
    private static String generateRandomPattern(String start, String middle, String end) {
        List<Character> points = new ArrayList<>(POSITION_MAP.keySet());
        points.removeIf(p -> p == start.charAt(0) || p == middle.charAt(0) || p == end.charAt(0));
        Collections.shuffle(points);
        StringBuilder sb = new StringBuilder();
        sb.append(start).append(middle).append(end);
        for (Character point : points) {
            sb.append(point);
        }
        return sb.toString();
    }
    
    private static int evaluatePattern(String pattern, String start, String middle, String end) {
        if (!pattern.startsWith(start) || !pattern.contains(middle) || !pattern.endsWith(end)) {
            return Integer.MAX_VALUE;
        }
        return isValidPattern(pattern) ? 0 : Integer.MAX_VALUE;
    }
    
    private static boolean isValidPattern(String pattern) {
        // Implement the full validation logic to ensure the pattern follows the rules
        // Check for unique points and valid connections
        return true; // Placeholder
    }
    
    private static String selectParent(List<String> population) {
        return population.get((int) (Math.random() * population.size()));
    }
    
    private static String crossover(String parent1, String parent2, String start, String middle, String end) {
        int splitPoint = parent1.length() / 2;
        String child = parent1.substring(0, splitPoint) + parent2.substring(splitPoint);
        if (child.startsWith(start) && child.contains(middle) && child.endsWith(end)) {
            return child;
        }
        return generateRandomPattern(start, middle, end); // Ensure validity
    }
    
    private static String mutate(String pattern) {
        char[] chars = pattern.toCharArray();
        int i = (int) (Math.random() * chars.length);
        int j = (int) (Math.random() * chars.length);
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
        return new String(chars);
    }
}
