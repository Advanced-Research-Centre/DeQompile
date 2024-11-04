OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2,q3,q4,q5 { p(pi/8) q0; p(pi/8) q1; cx q0,q1; p(-pi/8) q1; cx q0,q1; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; u(pi/2,pi/8,-pi) q5; cx q2,q5; p(-pi/8) q5; cx q1,q5; p(pi/8) q5; cx q2,q5; p(-pi/8) q5; cx q0,q5; p(pi/8) q5; cx q2,q5; p(-pi/8) q5; cx q1,q5; p(pi/8) q1; p(pi/8) q5; cx q2,q5; p(pi/8) q2; p(-pi/8) q5; cx q0,q5; p(pi/8) q0; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; u(pi/4,pi/2,-pi/2) q5; cx q4,q5; p(-pi/4) q5; cx q3,q5; p(pi/4) q5; cx q4,q5; p(pi/4) q4; p(-pi/4) q5; cx q3,q5; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q3,q4; u(pi/4,-3*pi/8,3*pi/4) q5; cx q2,q5; p(-pi/8) q5; cx q1,q5; p(pi/8) q5; cx q2,q5; p(-pi/8) q5; cx q0,q5; p(pi/8) q5; cx q2,q5; p(-pi/8) q5; cx q1,q5; p(pi/8) q5; cx q2,q5; p(-pi/8) q5; cx q0,q5; u(pi/4,pi/2,-pi/2) q5; cx q4,q5; p(-pi/4) q5; cx q3,q5; p(pi/4) q5; cx q4,q5; p(pi/4) q4; p(-pi/4) q5; cx q3,q5; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q3,q4; u(pi/2,pi/4,-3*pi/4) q5; u(pi/2,0,pi) q4; cx q1,q4; p(-pi/4) q4; cx q0,q4; p(pi/4) q4; cx q1,q4; p(pi/4) q1; p(-pi/4) q4; cx q0,q4; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/8,pi/2,-pi/4) q4; cx q3,q4; p(-pi/4) q4; cx q2,q4; p(pi/4) q4; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q2,q4; cx q2,q3; p(pi/4) q2; p(-pi/4) q3; cx q2,q3; u(pi/8,-pi/2,3*pi/4) q4; cx q1,q4; p(-pi/4) q4; cx q0,q4; p(pi/4) q4; cx q1,q4; p(pi/4) q1; p(-pi/4) q4; cx q0,q4; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/8,pi/2,-pi/4) q4; cx q3,q4; p(-pi/4) q4; cx q2,q4; p(pi/4) q4; cx q3,q4; p(pi/4) q3; p(-pi/4) q4; cx q2,q4; cx q2,q3; p(pi/4) q2; p(-pi/4) q3; cx q2,q3; u(pi/2,pi/8,-3*pi/4) q4; u(pi/2,0,pi) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/16,-3*pi/4) q3; cx q2,q3; u(pi/2,0,-15*pi/16) q3; cx q1,q3; p(-pi/4) q3; cx q0,q3; p(pi/4) q3; cx q1,q3; p(pi/4) q1; p(-pi/4) q3; cx q0,q3; cx q0,q1; p(pi/4) q0; p(-pi/4) q1; cx q0,q1; u(pi/2,-pi/16,-3*pi/4) q3; cx q2,q3; u(0,-3.043417883165112,-3.0434178831651124) q3; cx q0,q2; u(0,-pi/64,-pi/64) q2; cx q1,q2; u(0,-3.0925052683774528,-3.0925052683774528) q2; cx q0,q2; u(0,-pi/64,-pi/64) q2; cx q1,q2; u(0,-3.0925052683774528,-3.0925052683774528) q2; u(0,0,pi/32) q1; cx q0,q1; u(0,0,-pi/32) q1; cx q0,q1; p(pi/32) q0; }
gate rcccx q0,q1,q2,q3 { u2(0,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(0,pi) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; u2(0,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(0,pi) q3; }
gate rcccx_dg q0,q1,q2,q3 { u2(-2*pi,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(-2*pi,pi) q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u1(pi/4) q3; cx q1,q3; u1(-pi/4) q3; cx q0,q3; u2(-2*pi,pi) q3; u1(pi/4) q3; cx q2,q3; u1(-pi/4) q3; u2(-2*pi,pi) q3; }
gate mcx q0,q1,q2,q3,q4 { h q4; cu1(pi/2) q3,q4; h q4; rcccx q0,q1,q2,q3; h q4; cu1(-pi/2) q3,q4; h q4; rcccx_dg q0,q1,q2,q3; c3sqrtx q0,q1,q2,q4; }
gate mcx_2004275567872 q0,q1,q2,q3,q4 { mcx q0,q1,q2,q3,q4; }
gate gate_Q q0,q1,q2,q3,q4,q5 { mcphase(pi) q0,q1,q2,q3,q4,q5; h q4; h q3; h q2; h q1; h q0; x q0; x q1; x q2; x q3; x q4; h q4; mcx_2004275567872 q0,q1,q2,q3,q4; h q4; x q0; x q1; x q2; x q3; x q4; h q0; h q1; h q2; h q3; h q4; }
gate gate_Q_2004275568784 q0,q1,q2,q3,q4,q5 { gate_Q q0,q1,q2,q3,q4,q5; }
gate gate_Q_2004275569648 q0,q1,q2,q3,q4,q5 { gate_Q q0,q1,q2,q3,q4,q5; }
gate gate_Q_2004276015232 q0,q1,q2,q3,q4,q5 { gate_Q q0,q1,q2,q3,q4,q5; }
gate gate_Q_2004276870704 q0,q1,q2,q3,q4,q5 { gate_Q q0,q1,q2,q3,q4,q5; }
qreg q[5];
qreg flag[1];
creg meas[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
x flag[0];
gate_Q_2004275568784 q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q_2004275569648 q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q_2004276015232 q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q_2004276870704 q[0],q[1],q[2],q[3],q[4],flag[0];
barrier q[0],q[1],q[2],q[3],q[4],flag[0];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure flag[0] -> meas[5];