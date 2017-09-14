package com.github.chrbayer84.geneticlinearregression;

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;

public class GeneticLinearRegression implements Runnable {
    Set<Tuple> datapoints = new HashSet<>();
    int generations = 10;

    public GeneticLinearRegression() {
        datapoints.add(new Tuple(4.0, 6.0));
        datapoints.add(new Tuple(5.0, 8.0));
        datapoints.add(new Tuple(6.0, 10.0));
    }

    @Override
    public void run() {
        try {
            Population population = new Population(10, 0.2, 10);
            for (int i = 0; i < generations; i++) {
                population = population.reproduce(datapoints);
            }
            // sort results
            Set<Specimen> fitParents = population.sortParentsByFitness(datapoints);
            for (Specimen s : population.parents) {
                fitParents.add(s);
            }
            int i = 0;
            //print 5 best solutions
            for (Specimen s : fitParents) {
                if (i >= 5) {
                    break;
                }
                System.out.println("c0: " + s.t1 + " c1: " + s.t2 + " fitness: " + s.fitness);
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    class Population {
        int mutationPercentage;
        double mutationEffect;
        Set<Specimen> parents;

        public Population(int mutationPercentage, double mutationEffect, Set<Specimen> parents) {
            this.mutationPercentage = mutationPercentage;
            this.mutationEffect = mutationEffect;
            this.parents = parents;
        }

        public Population(int mutationPercentage, double mutationEffect, int populationSize) {
            this.mutationPercentage = mutationPercentage;
            this.mutationEffect = mutationEffect;
            this.parents = new HashSet<>();
            for (int i = 0; i < populationSize; i++)
                parents.add(new Specimen(ThreadLocalRandom.current().nextDouble(1, 10 + 1), ThreadLocalRandom.current().nextDouble(1, 10 + 1)));
        }

        private Specimen mutate(Specimen t) {
            return new Specimen(mutationEffect * t.t1, mutationEffect * t.t2);
        }

        // https://en.wikipedia.org/wiki/Genetic_algorithm
        Population reproduce(Set<Tuple> datapoints) throws Exception {
            Set<Specimen> fitParents = sortParentsByFitness(datapoints);

            // select fit parents
            int i = 0;
            List<Specimen> specimenList = new ArrayList<>();
            for (Specimen s : fitParents) {
                // half of all parents can reproduce and produce two children
                if (i >= fitParents.size() / 2) {
                    break;
                }
                specimenList.add(s);
                i++;
            }
            // combine
            Set<Specimen> children = new HashSet<>();
            for (int j = 1; j < specimenList.size(); j++) {
                List<Specimen> kid = crossover(specimenList.get(j), specimenList.get(j - 1));
                // mutate, only ever mutate childen1
                if (j - 1 % mutationPercentage == 0) {
                    kid.set(0, mutate(kid.get(0)));
                }
                for (Specimen k : kid) {
                    children.add(k);
                }
            }
            return new Population(mutationPercentage, mutationEffect, children);
        }

        public Set<Specimen> sortParentsByFitness(Set<Tuple> datapoints) throws Exception {
            // sort: fit parents are on top
            Set<Specimen> fitParents = new TreeSet<>((o1, o2) -> o1.fitness > o2.fitness ? 1 : -1);
            for (Specimen s : parents) {
                for (Tuple datapoint : datapoints) {
                    s.calculateFitness(datapoint.t1, datapoint.t2);
                }
                fitParents.add(s);
            }
            return fitParents;
        }

        private List<Specimen> crossover(Specimen specimen1, Specimen specimen2) {
            List<Specimen> result = new ArrayList<>();
            result.add(new Specimen(specimen1.t1, specimen2.t2));
            result.add(new Specimen(specimen2.t1, specimen1.t2));
            return result;
        }
    }

    class Tuple {
        double t1;
        double t2;

        public Tuple(double t1, double t2) {
            this.t1 = t1;
            this.t2 = t2;
        }
    }

    class Specimen extends Tuple {
        double fitness;

        public Specimen(double t1, double t2) {
            super(t1, t2);
        }

        // this can be a black box: given x, y, return some number that determines fitness, the bigger, the less fit
        void calculateFitness(double x, double y) throws Exception {
            double simple = new Error(t1, t2, x, y).call();
            // least squares: https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Motivational_example
            this.fitness += simple * simple;
        }
    }

    // can be given or a black box. Method with the signature double func(c0, c1, x)
    // could also be modeled as a lambda
    class Func implements Callable<Double> {
        double c0;
        double c1;
        double x;

        public Func(double c0, double c1, double x) {
            this.c0 = c0;
            this.c1 = c1;
            this.x = x;
        }

        @Override
        public Double call() throws Exception {
            return x * c1 + c0;
        }
    }

    // can be given or a black box. Method with the signature double error(c0, c1, x, y)
    // could also be modeled as a lambda
    class Error implements Callable<Double> {
        double c0;
        double c1;
        double x;
        double y;

        public Error(double c0, double c1, double x, double y) {
            this.c0 = c0;
            this.c1 = c1;
            this.x = x;
            this.y = y;
        }

        @Override
        public Double call() throws Exception {
            return y - new Func(c0, c1, x).call();
        }
    }

    public static void main(String[] args) {
        GeneticLinearRegression geneticLinearRegression = new GeneticLinearRegression();
        geneticLinearRegression.run();
    }
}
