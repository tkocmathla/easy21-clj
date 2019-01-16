(ns easy21.environment)

(defn draw []
  (* (inc (rand-int 10)) (rand-nth [-1 1 1])))

(defn play-dealer [sum]
  (cond
    ; bust
    (or (< sum 1) (> sum 21)) -1
    ; hit
    (< sum 17) (recur (+ sum (draw)))
    ; stick
    :else sum))

(defn step [{:keys [dealer player] :as state} action]
  (condp = action
    :hit (let [player (+ player (draw))]
           (if (or (< player 1) (> player 21))
             [::end -1]
             [(assoc state :player player) 0]))

    :stick (let [dealer-sum (play-dealer dealer)]
             (cond
               (or (not (<= 1 dealer-sum 21)) (> player dealer-sum))
               [::end 1]

               (= dealer-sum player)
               [::end 0]

               (< player dealer-sum)
               [::end -1]))))
