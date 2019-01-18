(ns easy21.sarsa
  "Sarsa(lambda) algorithm"
  (:require
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
;; gamma    discount factor
;; lambda   scaling factor

(defn sarsa
  [{:keys [S A N E Q alpha gamma lambda] :as m}]
  (let [[S* R] (env/step S A)
        A* (e-greedy S* Q N)
        error (+ R (* gamma (Q [S* A*] 0)) (- (Q [S A] 0)))]
    (-> m
        (update-in [:E [S A]] (fnil inc 0))
        (update-in [:N [S A]] (fnil inc 0))
        (update :Q (fn [Q] (reduce (fn [q s] (update q s (fnil + 0) (* alpha error (E s 0)))) Q env/all-states)))
        (update :E (fn [E] (reduce (fn [e s] (update e s (fnil * 1) gamma lambda)) E env/all-states)))
        (assoc :S S*)
        (assoc :A A*))))

(defn episode [init]
  (->> {:S [(Math/abs (env/draw)) (Math/abs (env/draw))]
        :A :hit
        :E {}
        :alpha 0.9
        :gamma 0.9
        :lambda 0.9}
       (merge init)
       (iterate sarsa)
       (drop-while (comp not #{::env/end} :S))
       first))

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
