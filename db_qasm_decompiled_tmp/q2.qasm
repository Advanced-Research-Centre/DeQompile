OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
ry(pi/23) q[0];
ry(pi/23) q[1];
ry(1.523196438104142) q[0];
ry(1.0053096491487339) q[1];
ry(pi/9) q[0];
ry(pi/13) q[1];
rx(4*pi/5) q[0];
rx(4*pi/5) q[1];
rx(2*pi/3) q[0];
rx(2*pi/3) q[1];
rx(2*pi/3) q[0];
h q[0];
h q[1];
rz(pi/4) q[1];
rz(pi/5) q[1];
ry(pi/6) q[0];
ry(pi/6) q[1];
ry(pi/6) q[0];
ry(pi/7) q[0];
ry(pi/7) q[1];
ry(pi/7) q[0];
ry(pi/7) q[1];