OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
rx(pi) q[0];
rx(pi/2) q[1];
rx(pi/4) q[2];
rx(pi/8) q[3];
rx(pi/16) q[4];
rx(pi/32) q[5];
rx(pi/64) q[6];
rx(pi/128) q[7];
rx(pi/256) q[8];
rx(pi/512) q[9];
rx(pi/1024) q[10];
rx(pi/2048) q[11];
rx(pi/4096) q[12];
rx(pi/8192) q[13];