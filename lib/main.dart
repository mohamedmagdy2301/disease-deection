// ignore_for_file: annotate_overrides, library_private_types_in_public_api, use_key_in_widget_constructors

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  TextEditingController ageController = TextEditingController();

  String predictionResult = '';

  Future<void> makePredictionRequest() async {
    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:5000/predict'),
        // Uri.parse('http://192.168.1.6:5000/predict'),
        headers: <String, String>{
          'Content-Type': 'application/json',
          // 'Access-Control-Allow-Origin': 'http://192.168.1.6:5000',
        },
        body: jsonEncode({
          'text': ageController.text.isNotEmpty ? ageController.text : " ",
        }),
      );
      if (ageController.text.isNotEmpty) {
        if (response.statusCode == 200) {
          final Map<String, dynamic> data = jsonDecode(response.body);
          setState(() {
            predictionResult = '${data['prediction']}';
          });
        } else {
          setState(() {
            predictionResult =
                'Failed to make a prediction request :${response.statusCode}';
          });
        }
      } else {
        setState(() {
          predictionResult = 'Please enter some text';
        });
      }
    } catch (e) {
      setState(() {
        predictionResult = 'Error: $e';
      });
    }
  }

  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection of diseases'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView(
          children: [
            const Text(
              'Enter the details of the patient:',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),

            TextField(
              controller: ageController,
              keyboardType: TextInputType.text,
              decoration: const InputDecoration(labelText: 'Description'),
            ),

            // Add more text fields for other input parameters

            const SizedBox(height: 20),

            ElevatedButton(
              onPressed: makePredictionRequest,
              child: const Text('Make Prediction'),
            ),

            const SizedBox(height: 60),

            Center(
              child: Text(
                predictionResult,
                style: const TextStyle(
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  color: Colors.red,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
