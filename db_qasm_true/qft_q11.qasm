OPENQASM 2.0;
include "qelib1.inc";
gate gate_QFT q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10 { h q10; cp(pi/2) q10,q9; cp(pi/4) q10,q8; cp(pi/8) q10,q7; cp(pi/16) q10,q6; cp(pi/32) q10,q5; cp(pi/64) q10,q4; cp(pi/128) q10,q3; cp(pi/256) q10,q2; cp(pi/512) q10,q1; cp(pi/1024) q10,q0; h q9; cp(pi/2) q9,q8; cp(pi/4) q9,q7; cp(pi/8) q9,q6; cp(pi/16) q9,q5; cp(pi/32) q9,q4; cp(pi/64) q9,q3; cp(pi/128) q9,q2; cp(pi/256) q9,q1; cp(pi/512) q9,q0; h q8; cp(pi/2) q8,q7; cp(pi/4) q8,q6; cp(pi/8) q8,q5; cp(pi/16) q8,q4; cp(pi/32) q8,q3; cp(pi/64) q8,q2; cp(pi/128) q8,q1; cp(pi/256) q8,q0; h q7; cp(pi/2) q7,q6; cp(pi/4) q7,q5; cp(pi/8) q7,q4; cp(pi/16) q7,q3; cp(pi/32) q7,q2; cp(pi/64) q7,q1; cp(pi/128) q7,q0; h q6; cp(pi/2) q6,q5; cp(pi/4) q6,q4; cp(pi/8) q6,q3; cp(pi/16) q6,q2; cp(pi/32) q6,q1; cp(pi/64) q6,q0; h q5; cp(pi/2) q5,q4; cp(pi/4) q5,q3; cp(pi/8) q5,q2; cp(pi/16) q5,q1; cp(pi/32) q5,q0; h q4; cp(pi/2) q4,q3; cp(pi/4) q4,q2; cp(pi/8) q4,q1; cp(pi/16) q4,q0; h q3; cp(pi/2) q3,q2; cp(pi/4) q3,q1; cp(pi/8) q3,q0; h q2; cp(pi/2) q2,q1; cp(pi/4) q2,q0; h q1; cp(pi/2) q1,q0; h q0; swap q0,q10; swap q1,q9; swap q2,q8; swap q3,q7; swap q4,q6; }
qreg q[11];
creg c[11];
creg meas[11];
gate_QFT q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];