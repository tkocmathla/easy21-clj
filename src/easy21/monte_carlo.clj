(ns easy21.monte-carlo
  (:require [easy21.environment :as env]))

(defn monte-carlo
  [{:keys [state counts history value] :as m}]
  (let [n (reduce + (vals (filter (comp #{state} first key) counts)))
        e (/ 100 (+ 100 n))
        action-vals (filter (comp #{state} first key) value)
        action (if (or (< (rand) e) (empty? action-vals))
                 (rand-nth [:hit :stick])
                 (second (key (apply max-key val action-vals))))
        [new-state reward] (env/step state action)]
    (-> m
        (assoc :state new-state)
        (update :reward + reward)
        (update :counts update [state action] (fnil inc 0))
        (update :history conj [state action]))))

(defn finish
  [{:keys [value counts history reward]}]
  [(reduce
     (fn [vf h]
       (let [a (/ 1 (counts h))]
         (assoc vf h (float (+ (vf h 0) (* a (- reward (vf h 0))))))))
     value history)
   counts])

(defn episode [[value counts]]
  (->> {:state {:dealer (Math/abs (env/draw))
                :player (Math/abs (env/draw))}
        :value value
        :counts counts
        :history []
        :reward 0}
       (iterate monte-carlo)
       (drop-while (comp not #{::env/end} :state))
       first
       finish))

;; -----------------------------------------------------------------------------

(comment
  (require
    '[clojure.pprint :refer [pprint]]
    '[com.stuartsierra.frequencies :as freq])

  (let [[vf] (->> [{} {}] (iterate episode) (take 10000) last)]
    ; dump summary of value function
    (pprint (-> vf vals frequencies freq/stats))
    ; dump states sorted by value
    (pprint (sort-by val > vf))))
