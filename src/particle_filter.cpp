/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;
    
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p;

    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1.0;

    particles.push_back(p);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
    std::default_random_engine gen;
    for(int i=0; i < num_particles; i++) {
        double delta_theta = delta_t*yaw_rate;
        if(abs(yaw_rate) > 0.00001) {
            particles[i].x += velocity/yaw_rate*(sin(particles[i].theta + delta_theta) - sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + delta_theta));
            particles[i].theta += delta_theta;
            std::normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
            std::normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
            std::normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
            particles[i].x = dist_x(gen);
            particles[i].y = dist_y(gen);
            particles[i].theta = dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    for (long unsigned int i=0; i < observations.size(); i++) {
        double min_dist = 100000000.0;
        double x_obs = observations[i].x;
        double y_obs = observations[i].y;
        for (long unsigned int j=0; j < predicted.size(); j++) {
            double distance = dist(predicted[j].x, predicted[j].y, x_obs, y_obs);
            if (distance < min_dist) {
                min_dist = distance;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    for(int i=0; i < num_particles; i++) {
        vector<LandmarkObs> predicted;
        vector<LandmarkObs> transformed(observations.size());
        for(long unsigned int k=0; k < map_landmarks.landmark_list.size(); k++) {
            // Find map landmark that is closest to predicted location
            LandmarkObs predicted_obs;
            double distance = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
            if (distance <= sensor_range) { 
                predicted_obs.x = map_landmarks.landmark_list[k].x_f;
                predicted_obs.y = map_landmarks.landmark_list[k].y_f;
                //predicted_obs.id = map_landmarks.landmark_list[k].id_i;
                predicted_obs.id = k; // Assign landmark index instead of id since that can be used to access
                predicted.push_back(predicted_obs);
            }
        }

        for(long unsigned int j=0; j < observations.size(); j++) {
            double x_transformed = 0.0;
            double y_transformed = 0.0;
            double angle = particles[i].theta;
            LandmarkObs transformed_obs;

            x_transformed = observations[j].x*cos(angle) + observations[j].y*sin(angle) + particles[i].x;
            y_transformed = -observations[j].x*sin(angle) + observations[j].y*cos(angle) + particles[i].y;
            transformed_obs.x = x_transformed;
            transformed_obs.y = y_transformed;
            transformed_obs.id = 0; //assign 0 because actual id is not known

            transformed.push_back(transformed_obs);
        }

        dataAssociation(predicted, transformed);

        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        for(long unsigned int j=0; j < transformed.size(); j++) {
            associations.push_back(transformed[j].id);
            sense_x.push_back(transformed[j].x);
            sense_y.push_back(transformed[j].y);
        }
        SetAssociations(particles[i], associations, sense_x, sense_y);

        particles[i].weight = 1.0;
        for(long unsigned int j=0; j < particles[i].associations.size(); j++) {
            double x = transformed[j].x;
            double y = transformed[j].y;
            double sigma_x = std_landmark[0];
            double sigma_y = std_landmark[1];
            double rho = 0.0;
            int index_landmark = particles[i].associations[j];
            double mu_x = map_landmarks.landmark_list[index_landmark].x_f;
            double mu_y = map_landmarks.landmark_list[index_landmark].y_f;
            particles[i].weight *= 0.5/(M_PI*sigma_x*sigma_y*(1.0-rho*rho))*exp(-0.5/(1.0-rho*rho)*(pow((x-mu_x)/sigma_x,2) + pow((y-mu_y)/sigma_y,2) - 2.0*rho*(x-mu_x)*(y-mu_y)/(sigma_x*sigma_y)));
        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    vector<double> weights;
    std::vector<Particle> resampled_particles;

    for(int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());

    for(int i=0; i < num_particles; i++) {
        resampled_particles.push_back(particles[d(gen)]);
    }

    for(int i=0; i < num_particles; i++) {
        particles[i] = resampled_particles[i];
    }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
