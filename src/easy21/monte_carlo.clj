(ns easy21.monte-carlo
  "Monte Carlo algorithm"
  (:require
    [easy21.environment :as env]
    [easy21.policies :refer [e-greedy]]))

;; A      action
;;        one of #{:hit :stick}
;; N      visit count function
;;        a map from [S A] -> number of visits over all episodes
;; Q      action value function
;;        a map from [S A] -> value
;; R      reward
;; S,S*   state, next state
;;        a 2-tuple of [dealer-card player-sum]
;; alpha  learning rate
;;        controls how far to shift predicted value toward actual value

(defn monte-carlo
  [{:keys [S Q N] :as m}]
  (let [A (e-greedy S Q N)
        [S* R] (env/step S A)]
    (-> m
        (assoc :S S*)
        (update :R + R)
        (update-in [:N [S A]] (fnil inc 0))
        (update :moves conj [S A]))))

(defn- finish
  [{:keys [Q N R moves] :as m}]
  (assoc m :Q (reduce
                (fn [Q move]
                  (let [alpha (/ 1.0 (N move 0))]
                    (update Q move (fnil + 0) (* alpha (- R (Q move 0))))))
                Q moves)))

(defn episode [init]
  (->> {:S [(Math/abs (env/draw)) (Math/abs (env/draw))]
        :R 0
        :moves []}
       (merge init)
       (iterate monte-carlo)
       (drop-while (comp not #{::env/end} :S))
       first
       finish))

;; -----------------------------------------------------------------------------

(comment
  (require
    '[clojure.pprint :refer [pprint]]
    '[com.stuartsierra.frequencies :as freq])

  (let [{:keys [Q]} (->> {:Q {} :N {}} (iterate episode) (take 1000) last)]
    ; dump summary of value function
    (pprint (-> Q vals frequencies freq/stats))
    ; dump states sorted by value
    (pprint (sort-by val > Q))))
