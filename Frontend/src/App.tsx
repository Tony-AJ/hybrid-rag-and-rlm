import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Nodes from './pages/Nodes'
import Edges from './pages/Edges'
import Search from './pages/Search'
import PDFUpload from './pages/PDFUpload'
import GraphView from './pages/GraphView'
import TestRunner from './pages/TestRunner'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/nodes" element={<Nodes />} />
        <Route path="/edges" element={<Edges />} />
        <Route path="/search" element={<Search />} />
        <Route path="/pdf" element={<PDFUpload />} />
        <Route path="/graph" element={<GraphView />} />
        <Route path="/tests" element={<TestRunner />} />
      </Routes>
    </Layout>
  )
}

export default App

