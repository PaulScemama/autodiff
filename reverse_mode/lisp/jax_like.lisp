

(defclass Node ()
    ((val
        :initarg :val
        :accessor val)
    
    (&optional (parents (list))
        :initarg :parents
        :accessor parents)
    
    (&optional (grad_fn (nil))
        :initarg :grad_fn
        :accessor grad_fn)
        ))


; (defvar node (make-instance 'Node :val 1.0 :parents (list 1 2 3) :grad_fn 1.0))

;; Addition operator
; (defun add_grad (g x y)
;     (list g g))

; (defun add (x y)
;     (defvar node (make-instance 
;     'Node 
;         :val (+ x y) 
;         :parents (list x y)
;         :grad_fn add_grad
;         )))

; ;; Subtraction operator
; (defun sub_grad (g x y)
;     (list g (- g)))

; (defun sub (x y)
;     (defvar node (make-instance 
;     'Node 
;         :val (- x y) 
;         :parents (list x y)
;         :grad_fn sub_grad
;         )))


; (defvar x (make-instance 'Node :val 1.0 :parents () :grad_fn ()))
; (defvar y (make-instance 'Node :val 2.0 :parents () :grad_fn ()))

; (add x y)

(defgeneric add (x y)
    (:documentation "Add two elements together."))

(defgeneric sub (x y)
    (:documentation "Subtract an element y from an element x."))

(defgeneric mul (x y)
    (:documentation "Multiply two elements together."))

(defgeneric div (x y)
    (:documentation "Divide an element y from an element x."))

(defgeneric pow (x y)
    (:documentation "Take the power y of an element x."))



(defmethod add ((x Node y Node))
    (+ )) 