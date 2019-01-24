(ns easy21.sarsa
  "Sarsa(lambda) algorithm"
  (:require
    [clojure.edn :as edn]
    [easy21.environment :as env]
    [easy21.policies :refer [e-greedy]]))

;; A,A*     action
;;          one of #{:hit :stick}
;; E,e      eligibility traces
;;          a map from [S A] -> discounted visit count
;; N        visit count function
;;          a map from [S A] -> number of visits over all episodes
;; Q,q      action value function
;;          a map from [S A] -> value
;; R        reward
;; S,S*,s   state
;;          a 2-tuple of [dealer-card player-sum]
;; alpha    learning rate
;;          controls how far to shift predicted value toward actual value
;; delta    error
;; gamma    discount factor
;; lambda   scaling factor

(defn sarsa
  [{:keys [S A N Q gamma lambda] :as m}]
  (let [[S* R] (env/step S A)
        A* (e-greedy S* Q N)
        delta (+ R (* gamma (Q [S* A*] 0)) (- (Q [S A] 0)))
        {:keys [E N moves] :as new-m}
        (-> m
            (update-in [:E [S A]] (fnil inc 0))
            (update-in [:N [S A]] (fnil inc 0))
            (update :moves conj [S A])
            (assoc :S S*)
            (assoc :A A*))]
    (reduce
      (fn [m move]
        (let [alpha (/ 1.0 (N move))]
          (-> m
              (update-in [:Q move] (fnil + 0) (* alpha delta (E move)))
              (update-in [:E move] * gamma lambda))))
      new-m moves)))

(defn episode [init]
  (->> {:S [(Math/abs (env/draw)) (Math/abs (env/draw))]
        :A :hit
        :E {}
        :moves []
        :gamma 0.9}
       (merge init)
       (iterate sarsa)
       (drop-while (comp not #{::env/end} :S))
       first))

;; -----------------------------------------------------------------------------

#_
(comment
  (require '[clojure.string :as string])
  (set! *print-length* nil)

  (let [episodes (int 1e6)
        Q* (edn/read-string (slurp "Q.edn"))]
    (for [lambda [0.4] #_(range 0 11/10 1/10)
          :let [{:keys [Q]} (->> {:Q {} :N {} :lambda lambda} (iterate episode) (take episodes) last)]]

      (do (prn :MSE (/ (reduce + (map (fn [k] (Math/pow (- (Q k) (Q* k)) 2)) (keys Q)))
                       (count env/all-states)))

          (->> Q
               (map (fn [[[s a] q]] [s q]))
               (group-by first)
               (map (fn [[[d p] xs]] [d p (->> xs (map last) (apply max))]))
               (map #(string/join #"," %))
               (string/join "\n")
               doall
               (spit (format "Q-td-%se-%sl.csv" episodes (double lambda))))))))
