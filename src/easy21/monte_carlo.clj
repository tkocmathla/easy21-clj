(ns easy21.monte-carlo
  "Monte Carlo algorithm"
  (:require
    [easy21.environment :as env]
    [easy21.policies :refer [e-greedy]]))

;; A      action
;;        one of #{:hit :stick}
;; G      cumulative episode reward
;; N      visit count function
;;        a map from [S A] -> number of visits over all episodes
;; Q      action value function
;;        a map from [S A] -> value
;; r      step reward
;; S,S*   state, next state
;;        a 2-tuple of [dealer-card player-sum]
;; alpha  learning rate
;;        controls how far to shift predicted value toward actual value

(defn monte-carlo
  [{:keys [S Q N] :as m}]
  (let [A (e-greedy S Q N)
        [S* R] (env/step S A)]
    (-> m
        (update :SAR conj [S A R])
        (assoc :S S*))))

(defn finish
  [{:keys [SAR] :as m}]
  (let [G (reduce + (map last SAR))]
    (reduce
      (fn [{:keys [Q N] :as m} [s a r]]
        (let [alpha (/ 1 (inc (or (N [s a]) 0)))
              q (or (Q [s a]) (rand))]
          (-> m
              (update-in [:N [s a]] (fnil inc 0))
              (assoc-in [:Q [s a]] (+ q (* alpha (- r q)))))))
      m SAR)))

(defn episode [init]
  (->> {:S [(Math/abs (env/draw)) (Math/abs (env/draw))]
        :SAR []}
       (merge init)
       (iterate monte-carlo)
       (drop-while (comp not #{::env/end} :S))
       first
       finish))

;; -----------------------------------------------------------------------------

(comment
  (require '[clojure.string :as string])
  (set! *print-length* nil)

  (time
    (let [{:keys [Q]} (->> {:Q {} :N {}} (iterate episode) (take 5e6) last)]
      ; dump the optimal value function
      (spit "Q.edn" (pr-str Q))

      ; dump csv data for plotting in python
      (->> Q
           (map (fn [[[s a] q]] [s q]))
           (group-by first)
           (map (fn [[[d p] xs]] [d p (->> xs (map last) (apply max))]))
           (map #(string/join #"," %))
           (string/join "\n")
           doall
           (spit "Q-mc-5e6.csv")))))
