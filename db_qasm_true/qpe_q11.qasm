OPENQASM 2.0;
include "qelib1.inc";
gate gate_IQFT_dg q0,q1,q2,q3,q4,q5,q6,q7,q8,q9 { swap q4,q5; swap q3,q6; swap q2,q7; swap q1,q8; swap q0,q9; h q0; cp(-pi/2) q1,q0; h q1; cp(-pi/4) q2,q0; cp(-pi/2) q2,q1; h q2; cp(-pi/8) q3,q0; cp(-pi/4) q3,q1; cp(-pi/2) q3,q2; h q3; cp(-pi/16) q4,q0; cp(-pi/8) q4,q1; cp(-pi/4) q4,q2; cp(-pi/2) q4,q3; h q4; cp(-pi/32) q5,q0; cp(-pi/16) q5,q1; cp(-pi/8) q5,q2; cp(-pi/4) q5,q3; cp(-pi/2) q5,q4; h q5; cp(-pi/64) q6,q0; cp(-pi/32) q6,q1; cp(-pi/16) q6,q2; cp(-pi/8) q6,q3; cp(-pi/4) q6,q4; cp(-pi/2) q6,q5; h q6; cp(-pi/128) q7,q0; cp(-pi/64) q7,q1; cp(-pi/32) q7,q2; cp(-pi/16) q7,q3; cp(-pi/8) q7,q4; cp(-pi/4) q7,q5; cp(-pi/2) q7,q6; h q7; cp(-pi/256) q8,q0; cp(-pi/128) q8,q1; cp(-pi/64) q8,q2; cp(-pi/32) q8,q3; cp(-pi/16) q8,q4; cp(-pi/8) q8,q5; cp(-pi/4) q8,q6; cp(-pi/2) q8,q7; h q8; cp(-pi/512) q9,q0; cp(-pi/256) q9,q1; cp(-pi/128) q9,q2; cp(-pi/64) q9,q3; cp(-pi/32) q9,q4; cp(-pi/16) q9,q5; cp(-pi/8) q9,q6; cp(-pi/4) q9,q7; cp(-pi/2) q9,q8; h q9; }
qreg q[10];
qreg psi[1];
creg c[10];
x psi[0];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
cp(-2.693670263527186) psi[0],q[0];
cp(0.8958447801252144) psi[0],q[1];
cp(1.7916895602504288) psi[0],q[2];
cp(-2.6998061866787286) psi[0],q[3];
cp(0.8835729338221293) psi[0],q[4];
cp(9*pi/16) psi[0],q[5];
cp(-7*pi/8) psi[0],q[6];
cp(pi/4) psi[0],q[7];
cp(pi/2) psi[0],q[8];
cp(pi) psi[0],q[9];
gate_IQFT_dg q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],psi[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];