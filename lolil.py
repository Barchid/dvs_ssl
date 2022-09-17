'''
	Author: Jallson Suryo & Nicholas Patrick
	Date: 2022-07-25
	License: CC0
	Source: Edge Impulse python SDK example file (classify.py) -- modified
	Description: Program to count livestock or wildlife from a drone (moving camera) using
		Edge Impulse FOMO trained model.
'''
#!/usr/bin/env python

import device_patches     # Device specific patches for Jetson Nano (needs to be before importing cv2)

from math import inf, sqrt
from queue import Queue
import cv2
import os
import sys, getopt
import signal
import time
from edge_impulse_linux.image import ImageImpulseRunner

runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
	show_camera = False

def now():
	return round(time.time() * 1000)

def get_webcams():
	port_ids = []
	for port in range(5):
		print("Looking for a camera in port %s:" %port)
		camera = cv2.VideoCapture(port)
		if camera.isOpened():
			ret = camera.read()[0]
			if ret:
				backendName =camera.getBackendName()
				w = camera.get(3)
				h = camera.get(4)
				print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
				port_ids.append(port)
			camera.release()
	return port_ids

def sigint_handler(sig, frame):
	print('Interrupted')
	if (runner):
		runner.stop()
	sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
	print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
	try:
		opts, args = getopt.getopt(argv, "h", ["--help"])
	except getopt.GetoptError:
		help()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			help()
			sys.exit()

	if len(args) == 0:
		help()
		sys.exit(2)

	model = args[0]

	dir_path = os.path.dirname(os.path.realpath(__file__))
	modelfile = os.path.join(dir_path, model)

	print('MODEL: ' + modelfile)

	with ImageImpulseRunner(modelfile) as runner:
		try:
			model_info = runner.init()
			print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
			labels = model_info['model_parameters']['labels']
			if len(args)>= 2:
				videoCaptureDeviceId = int(args[1])
			else:
				port_ids = get_webcams()
				if len(port_ids) == 0:
					raise Exception('Cannot find any webcams')
				if len(args)<= 1 and len(port_ids)> 1:
					raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
				videoCaptureDeviceId = int(port_ids[0])

			camera = cv2.VideoCapture(videoCaptureDeviceId)
			ret = camera.read()[0]
			if ret:
				backendName = camera.getBackendName()
				w = camera.get(3)
				h = camera.get(4)
				print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
				camera.release()
			else:
				raise Exception("Couldn't initialize selected camera.")

			HEIGHT = 96
			WIDTH = 96

			next_frame_start_time = 0
			prev_frame_objects = []
			cumulative_counts = {'duck' : 0, 'turtle' : 0}

			# iterate through frames
			for res, img in runner.classifier(videoCaptureDeviceId):
				# print('classification runner response', res)

				if "classification" in res["result"].keys():
					print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
					for label in labels:
						score = res['result']['classification'][label]
						print('%s: %.2f\t' % (label, score), end='')
					print('', flush=True)

				elif "bounding_boxes" in res["result"].keys():
					curr_frame_objects = res["result"]["bounding_boxes"]
					m, n = len(prev_frame_objects), len(curr_frame_objects)
					print('Found %d bounding boxes (%d ms.)' % (n, res['timing']['dsp'] + res['timing']['classification']))
					# iterate through identified objects
					for bb in curr_frame_objects:
						print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
						img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

					# Pairs objects seen in both the previous frame and the current frame.
					# To get a good pairing, each potential pair is given a cost. The problem
					# then transforms into minimum cost maximum cardinality bipartite matching.

					# populate table
					def get_c(a0, a1):
						# computes cost of pairs. A cost of inf implies no edge.
						A, B = sqrt(HEIGHT ** 2 + WIDTH ** 2) / 8, 5
						if a0['label'] != a1['label']: return inf
						d2 = (a0['x'] - a1['x']) ** 2 + (a0['x'] - a1['x']) ** 2
						dn4 = d2 ** -2 if d2 else 10**20
						val = a0['value'] * a1['value'] * (((1 + B) * dn4) / (dn4 + A ** -4) - B)
						return inf if val <= 0 else 1 - val
					match_c = [[get_c(i, j) for j in curr_frame_objects] for i in prev_frame_objects]

					# solves the matching problem in O(V^2E) by repeatedly finding augmenting paths
					# using shortest path faster algorithm (SPFA).
					# A modified Hungarian algorithm could also have been used.
					# 0..m-1: prev, left
					# m..m+n-1: this, right
					# m+n: source
					# m+n+1: sink
					source, sink, V = m + n, m + n + 1, m + n + 2
					matched = [-1] * (m + n + 2)
					adjLis = [[] for i in range(m)] + [[(sink, 0)] for _ in range(n)] + [[(i, 0) for i in range(m)], []]
					#        left                     right                              source                     sink
					for i in range(m):
						for j in range(n):
							if match_c[i][j] != inf:
								adjLis[i].append((j + m, match_c[i][j]))

					# finds augmenting paths until no more are found.
					while True:
						# SPFA
						distance = [inf] * V
						distance[source] = 0
						parent = [-1] * V
						Q, inQ = Queue(), [False] * V
						Q.put(source); inQ[source] = True
						while not Q.empty():
							u = Q.get(); inQ[u] = False
							for v, w in adjLis[u]:
								if u < m and matched[u] == v: continue
								if u == source and matched[v] != -1: continue
								if distance[u] + w < distance[v]:
									distance[v] = distance[u] + w
									parent[v] = u
									if not inQ[v]: Q.put(v); inQ[v] = True
						aug = parent[sink]
						if aug == -1: break
						# augment the shortest path
						while aug != source:
							v = aug
							aug = parent[aug]
							u = aug
							aug = parent[aug]
							adjLis[v] = [(u, -match_c[u][v - m])]
							matched[u], matched[v] = v, u

					# updating cumulative_counts by the unmatched new objects
					for i in range(n):
						if matched[m + i] == -1:
							cumulative_counts[curr_frame_objects[i]['label']] += 1

					# preparing prev_frame_objects for the next frame
					next_prev_frame_objects = curr_frame_objects
					# considering objects that became invisible (false negative) for a few frames.
					for i in range(m):
						if matched[i] != -1: continue
						prev_frame_objects[i]['value'] *= 0.7
						if prev_frame_objects[i]['value'] >= 0.35: 
							next_prev_frame_objects.append(prev_frame_objects[i])
					prev_frame_objects = next_prev_frame_objects

					print("current cumulative_counts:\n %d ducks, %d turtles" % (cumulative_counts['duck'], cumulative_counts['turtle']))

				if (show_camera):
					cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
					if cv2.waitKey(1) == ord('q'):
						break

				if (next_frame_start_time > now()):
					time.sleep((next_frame_start_time - now()) / 1000)
				# operates at a maximum of 5fps
				next_frame_start_time = now() + 200
		finally:
			if (runner):
				runner.stop()

if __name__ == "__main__":
	main(sys.argv[1:])
